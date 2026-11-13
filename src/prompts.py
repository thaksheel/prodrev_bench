import threading
import zlib
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd        
from sklearn.neighbors import NearestNeighbors


INSTRUCTION = (
    "Given this product review from a customer, what is the sentiment conveyed? "
    "Rate it from 1, 2, 3, 4, 5 where 1 is negative sentiment and 5 is positive sentiment."
)
ANSWER_ONLY = (
    "Return only the rating as a single number from 1 to 5, with no other text."
)
ANSWER_COT = (
    "Think through the review step by step, then answer with a JSON object: "
    '{"reasoning": "<your step-by-step reasoning>", "rating": <integer from 1 to 5>}'
)

Example = Tuple[str, int]


class RandomExamples:
    """Few-shot examples drawn at random from a pool of labelled reviews.

    Parameters
    ----------
    pool : pandas.DataFrame
        Labelled reviews to draw from.
    num_examples : int, default 10
        Examples per prompt.
    seed : int, default 0
        Combined with each review's text, so a review always gets the same
        examples for a given seed, whatever order reviews are processed in.
    review_col : str, default "review"
    rating_col : str, default "rating"
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        num_examples: int = 10,
        seed: int = 0,
        review_col: str = "review",
        rating_col: str = "rating",
    ):
        self.pool = pool.reset_index(drop=True)
        self.num_examples = num_examples
        self.seed = seed
        self.review_col = review_col
        self.rating_col = rating_col

    def prepare(self, reviews: Sequence[str]) -> None:
        pass

    def select(self, review: str) -> List[Example]:
        candidates = self.pool[self.pool[self.review_col] != review]
        state = (
            self.seed * 1_000_003 + zlib.crc32(str(review).encode("utf-8"))
        ) % 2**32
        rows = candidates.sample(
            n=min(self.num_examples, len(candidates)), random_state=state
        )
        return list(zip(rows[self.review_col], rows[self.rating_col].astype(int)))


class BertEncoder:
    """Mean-pooled sentence embeddings from a BERT-style model.

    The model is loaded once, texts are embedded in batches, and every
    embedding is cached by text, so repeated reviews are never re-embedded.

    Parameters
    ----------
    model_path : str, default "./transformer_models/bert"
        Local path or Hugging Face id of the encoder.
    device : str, optional
        "cuda" or "cpu". Defaults to CUDA when available.
    batch_size : int, default 64
    max_length : int, default 512
    """

    def __init__(
        self,
        device: torch.device,
        model_path: str = "./transformer_models/bert",
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device).eval()
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        with self._lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                for text, vector in zip(batch, self._embed(batch)):
                    self._cache[text] = vector
            return np.stack([self._cache[t] for t in texts])

    def _embed(self, batch: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)
        with self._torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.cpu().numpy()


class KATEExamples:
    """Few-shot examples nearest to the review in embedding space (KATE).

    The pool is embedded and indexed once; each prompt then only embeds the
    review being rated.

    Parameters
    ----------
    pool : pandas.DataFrame
        Labelled reviews to draw from.
    encoder : BertEncoder
        Shared encoder; create it once and reuse it.
    num_examples : int, default 10
        Examples per prompt, nearest first.
    review_col : str, default "review"
    rating_col : str, default "rating"
    """

    def __init__(
        self,
        pool: pd.DataFrame,
        encoder: BertEncoder,
        num_examples: int = 10,
        review_col: str = "review",
        rating_col: str = "rating",
    ):

        self.pool = pool.reset_index(drop=True)
        self.encoder = encoder
        self.review_col = review_col
        self.rating_col = rating_col
        embeddings = encoder.encode(self.pool[review_col].tolist())
        self.index = NearestNeighbors(
            n_neighbors=min(num_examples, len(self.pool))
        ).fit(embeddings)

    def prepare(self, reviews: Sequence[str]) -> None:
        self.encoder.encode(reviews)

    def select(self, review: str) -> List[Example]:
        _, idx = self.index.kneighbors(self.encoder.encode([review]))
        rows = self.pool.iloc[idx[0]]
        return list(zip(rows[self.review_col], rows[self.rating_col].astype(int)))


class PromptBuilder:
    """Chat messages for zero-shot or few-shot rating prediction.

    Parameters
    ----------
    strategy : {"zero-shot", "few-shots"}
    chain_of_thought : bool, default False
        False: the model must answer with the rating number only.
        True: the model reasons step by step and answers with a JSON object
        holding its reasoning and rating.
    examples : RandomExamples or KATEExamples, optional
        Example selector. Required for "few-shots".
    """

    def __init__(
        self,
        strategy: Literal["zero-shot", "few-shots"],
        chain_of_thought: bool = False,
        examples: Optional[Union[RandomExamples, KATEExamples]] = None,
    ):
        if strategy == "few-shots" and examples is None:
            raise ValueError("few-shots needs an example selector")
        self.strategy = strategy
        self.chain_of_thought = chain_of_thought
        self.examples = examples

    def prepare(self, reviews: Sequence[str]) -> None:
        """Embed all reviews up front (KATE), so prompts are built quickly later."""
        if self.examples is not None:
            self.examples.prepare(reviews)

    def messages(self, review: str) -> List[Dict[str, str]]:
        parts = [INSTRUCTION, ANSWER_COT if self.chain_of_thought else ANSWER_ONLY]
        if self.strategy == "few-shots":
            shots = "\n\n".join(
                f"Review: {text}\nRating: {rating}"
                for text, rating in self.examples.select(review)
            )
            parts.append("Here are some examples to learn from:\n\n" + shots)
        if self.chain_of_thought:
            user = f"Review: {review}"
        else:
            user = f"Review: {review}\nRating:"
        return [
            {"role": "system", "content": "\n\n".join(parts)},
            {"role": "user", "content": user},
        ]
