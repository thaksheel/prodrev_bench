import pandas as pd
from dataclasses import dataclass
from typing import TypedDict, Literal, Optional, Dict, List
from enum import Enum
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from transformers import BertModel, BertTokenizer

import warnings

warnings.filterwarnings("ignore")


class Methods(Enum):
    ZERO_SHOT = 0
    ZERO_SHOT_COT = 1
    FEW_SHOT = 2
    FEW_SHORT_COT = 3


class ClassLevels(Enum):
    BINARY = "-1 to 1 where -1 is negative sentiment and 1 is positive sentiment"
    THREE_LEVELS = "-1, 0, 1 where -1 is negative sentiments, 0 is neutral sentiments, and 1 is positive sentiments"
    FIVE_LEVELS = (
        "1, 2, 3, 4, 5 where 1 is negative sentiments and 5 is positive sentiments"
    )


class RatingReasoning(Enum):
    RATING = "Return only the rating as a number, nothing else."
    REASONING = "Return the response in the following format."
    JSON_RATING = """You will output a json object containing the following {rating: int // an integer based on the given sentiment level}"""
    JSON_REASONING = "You will output a json object containing the following {reasoning: string // a summary of the reasons why you gave the rating for this product based on the reviews, rating: int // an integer based on the given sentiment level, }"


class ShotsExamples:
    """Using random selections of demonstration through the df without current review"""

    def __init__(
        self,
        df: pd.DataFrame,
        review: str,
        review_col: str = "review",
        rating_col: str = "rating",
    ):
        self.review_col = review_col
        self.rating_col = rating_col
        self.df = df[df[review_col] != review]
        self.examples = []

    def get_examples(
        self,
        amount: int,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ) -> list:
        df_rand = self.df.sample(n=amount).reset_index(drop=True, inplace=False)
        examples = []
        for _, row in df_rand.iterrows():
            if demonstration and np.random.rand() < demon_rate:
                rating = np.random.random_integers(1, 5)
            else:
                rating = row[self.rating_col]
            review = row[self.review_col]
            r = (f"Rating: {int(rating)}", f"Review: {review}")
            examples.append(r)
        self.examples = examples
        return self.examples[:amount]


class KATEShotsExamples:
    """Using KATE with RoBERTa pretraiend on SST2 and kNN to select nearest neighbor of given review in entire dataset"""

    def __init__(
        self,
        df: pd.DataFrame,
        review: str,
        device: torch.device = None,
        embeddings: Optional[np.ndarray] = None,
        model_path: str = "./transformer_models/bert",
        cache_dir: str = "/Research/huggingface/transformers/cached",
        review_col: str = "review",
        rating_col: str = "rating",
    ):

        self.review_col = review_col
        self.rating_col = rating_col
        self.review = review
        # self.df = df[df[review_col] != review]
        self.df = df
        self.df.reset_index(inplace=True, drop=True)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)

        if embeddings is None:
            self.model.eval()
            self.df["embeddings"] = self.df[review_col].apply(self.get_review_embedding)
            self.embeddings = np.stack(self.df["embeddings"].values)
        elif embeddings is not None:
            self.embeddings = embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (summed / counts).squeeze(0).detach().cpu().numpy()

    def get_review_embedding(self, text) -> np.ndarray:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self.mean_pooling(outputs, inputs["attention_mask"])
        return embedding

    def get_examples(
        self,
        num_examples: int,
        review: Optional[str] = None,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ):
        if review is None:
            review = self.review
        review_embd = self.get_review_embedding(review).reshape(1, -1)
        nbrs = NearestNeighbors(
            n_neighbors=num_examples + 1, algorithm="ball_tree", n_jobs=-1
        ).fit(self.embeddings)
        distances, indices = nbrs.kneighbors(review_embd)
        few_shots = [
            (self.df.loc[i, self.review_col], int(self.df.loc[i, self.rating_col]))
            for k, i in enumerate(indices[0])
            if distances[0][k] > 0
        ]
        examples = []
        for rr in few_shots:
            if demonstration and np.random.rand() < demon_rate:
                rating = np.random.random_integers(1, 5)
            else:
                rating = rr[1]
            review = rr[0]
            r = (f"Rating: {int(rating)}", f"Review: {review}")
            examples.append(r)
        self.examples = examples
        return self.examples[:num_examples]

    def get_embeddings(self):
        return self.embeddings


class Responses(TypedDict):
    rating: int
    reasoning: str


@dataclass
class Prompt:
    review: str
    chain_of_thought: bool = False
    rreasoning: RatingReasoning = RatingReasoning.RATING
    class_level: ClassLevels = ClassLevels.FIVE_LEVELS
    num_examples: int = 10


class PromptStrategy:
    def __init__(
        self,
        prompt: Prompt,
        strategy: Literal["zero-shot", "few-shots", "re-prompt"],
        df: pd.DataFrame,
        few_shots_type: Literal["random", "kate"],
        responses: Optional[set[Responses]] = None,
        embeddings: Optional[np.ndarray] = None,
    ):
        self.prompt = prompt
        self.responses = responses
        self.strategy = strategy
        self.embeddings = embeddings
        self.few_shots_type = few_shots_type
        self.df = df

        # null fields
        self.kate: KATEShotsExamples = None

    def set_review(self, review):
        self.prompt.review = review
        return self

    def get_review(self):
        return self.prompt.review

    def zero_shot(self) -> Dict:
        if self.prompt.chain_of_thought:
            cot = "Let's work this out in a step by step way to ensure we have the right answer."
        else:
            cot = ""
        out_system = f"""Given this product review from a customer, what is the sentiment conveyed? Rate it from {self.prompt.class_level.value}? \n{cot} \n{self.prompt.rreasoning.value}"""

        out_user = f"""Review: {self.prompt.review} \nRating:"""
        return {
            "system": out_system,
            "user": out_user,
        }

    def few_shots(
        self,
        df_samples: pd.DataFrame,
        fewshots_samples: int = 500,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ) -> Dict:
        if self.prompt.chain_of_thought:
            cot = "Let's work this out in a step by step way to ensure we have the right answer."
        else:
            cot = ""
        examples_str = ""
        if self.few_shots_type == "random":
            shots = ShotsExamples(df_samples, self.prompt.review)
            rand_shots = shots.get_examples(
                self.prompt.num_examples,
                demonstration=demonstration,
                demon_rate=demon_rate,
            )
            for l in rand_shots:
                examples_str += f"{l[1]} \n{l[0]} \n"
        elif self.few_shots_type == "kate":
            if self.embeddings is not None:
                embeddings = self.embeddings
            else:
                embeddings = None
            self.kate = KATEShotsExamples(
                df_samples, self.prompt.review, embeddings=embeddings
            )
            self.embeddings = self.kate.get_embeddings()
            for l in self.kate.get_examples(
                self.prompt.num_examples,
                review=self.prompt.review,
                demon_rate=demon_rate,
                demonstration=demonstration,
            ):
                examples_str += f"{l[1]} \n{l[0]} \n"
        out_system = f"""Given this product review from a customer, what is the sentiment conveyed? Rate it from {self.prompt.class_level.value}. \n{cot} \n{self.prompt.rreasoning.value} \nHere are some examples to learn from: \n{examples_str}"""
        out_user = f"""Review: {self.prompt.review} \nRating: 
        """
        return {
            "system": out_system,
            "user": out_user,
        }

    def re_prompt(self) -> Optional[str]:
        res = ""
        if self.responses is not None:
            for i, z in enumerate(self.responses):
                res += f"assistant{i+1} rating={z['rating']} reasoning is {z['reasoning']}, \n"
            ins = f"""Given this product review from a customer, what is the sentiment conveyed? Rate it from {self.prompt.class_level.value}? 
            Product Review: {self.prompt.review}"""
            s = f"""You are a marketing sentiment analysis manager. Several sentiment analysis assistants gave were given the following instruction: {ins}. The assistants ratings are different. Consider each of their reasoning and select the rating that matches the product review. 
            Assistants reasoning: 
                {res}
            Return only the rating as a number, nothing else. The final rating is"""
            return s

    def get_prompt(
        self,
        df_samples: pd.DataFrame,
        fewshots_samples: int = 500,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ) -> Optional[str]:
        if self.strategy == "zero-shot":
            obj = self.zero_shot()
            out = obj["system"] + obj["user"]
            return out
        elif self.strategy == "few-shots":
            obj = self.few_shots(
                df_samples,
                demonstration=demonstration,
                fewshots_samples=fewshots_samples,
                demon_rate=demon_rate,
            )
            out = obj["system"] + obj["user"]
            return out
        elif self.strategy == "re-prompt":
            return self.re_prompt()
        else:
            return None

    def get_system_prompt(
        self,
        df_samples: pd.DataFrame,
        fewshots_samples: int = 500,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ) -> Optional[str]:
        if self.strategy == "zero-shot":
            return self.zero_shot()["system"]
        elif self.strategy == "few-shots":
            return self.few_shots(
                df_samples,
                demonstration=demonstration,
                fewshots_samples=fewshots_samples,
                demon_rate=demon_rate,
            )["system"]
        elif self.strategy == "re-prompt":
            return self.re_prompt()
        else:
            return None

    def get_user_prompt(
        self,
        df_samples: pd.DataFrame,
        fewshots_samples: int = 500,
        demonstration: bool = False,
        demon_rate: float = 0.5,
    ):
        if self.strategy == "zero-shot":
            return self.zero_shot()["user"]
        elif self.strategy == "few-shots":
            return self.few_shots(
                df_samples,
                demonstration=demonstration,
                fewshots_samples=fewshots_samples,
                demon_rate=demon_rate,
            )["user"]
        elif self.strategy == "re-prompt":
            return self.re_prompt()
        else:
            return None
