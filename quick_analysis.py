import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate_ratings(df, true_col="rating", pred_col="results"):
    """
    Evaluate 1-5 rating predictions at:
        1. 5-class classification
        2. 3-class classification
        3. Binary classification

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ground-truth and predicted ratings.

    true_col : str
        Column containing the ground-truth 1-5 ratings.

    pred_col : str
        Column containing the predicted 1-5 ratings.

    Returns
    -------
    pd.DataFrame
        Accuracy, weighted F1, and macro F1 for each classification level.
    """

    y_true = df[true_col]
    y_pred = df[pred_col]
    y_true_5 = y_true
    y_pred_5 = y_pred

    mapping_3 = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
    y_true_3 = y_true.map(mapping_3)
    y_pred_3 = y_pred.map(mapping_3)

    mapping_binary = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}
    y_true_binary = y_true.map(mapping_binary)
    y_pred_binary = y_pred.map(mapping_binary)

    def calculate_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            # Weighted by number of samples in each class
            "f1": f1_score(y_true, y_pred, average=None, zero_division=0),
            # Each class contributes equally
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    metrics_5 = calculate_metrics(y_true_5, y_pred_5)
    metrics_3 = calculate_metrics(y_true_3, y_pred_3)
    metrics_binary = calculate_metrics(y_true_binary, y_pred_binary)
    result_df = pd.DataFrame(
        [
            {"classification": "5-level", **metrics_5},
            {"classification": "3-level", **metrics_3},
            {"classification": "binary", **metrics_binary},
        ]
    )

    return result_df


df = pd.read_excel("./exports/rslt_openai_madp.xlsx")
df_results = evaluate_ratings(df)
df_results["len"] = [len(df)] * len(df_results)

print(df_results)
print("END")
