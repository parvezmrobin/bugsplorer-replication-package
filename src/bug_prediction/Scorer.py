from functools import cache
from typing import Iterable, NamedTuple

import numpy as np
from sklearn import metrics


def recall_at_20_percent_loc(y_true: np.ndarray, y_score: np.ndarray):
    twenty_percent = int(len(y_true) * 0.20)
    # find the indices with the highest prediction value
    y_score_sorted_idx = np.argsort(y_score)[::-1]
    # find the index of top 20% prediction
    top_twenty_percent_index = y_score_sorted_idx[:twenty_percent]
    # find true value for the top 20% prediction
    top_twenty_percent = y_true[top_twenty_percent_index]
    # get what fraction of them are actually true
    result = np.sum(top_twenty_percent) / np.sum(y_true)
    return result


def effort_at_20_percent_recall(y_true: np.ndarray, y_score: np.ndarray):
    twenty_percent_defect = int(sum(y_true) * 0.20)
    # find the indices with the highest prediction value
    y_score_sorted_idx = np.argsort(y_score)[::-1]

    defect_count = 0
    for i, j in enumerate(y_score_sorted_idx, 1):
        if y_true[j]:
            defect_count += 1
        if defect_count == twenty_percent_defect:
            return i / len(y_true)

    return 1


def initial_false_alarm(y_true: np.ndarray, y_pred: np.ndarray):
    true_positives = (y_true == True) & (y_pred == True)
    true_positives[-1] = True
    assert all(map(lambda x: x in [True, False], true_positives))
    return true_positives.tolist().index(True) / len(true_positives)


custom_metrics = {
    "recall_at_20_percent_loc": recall_at_20_percent_loc,
    "effort_at_20_percent_recall": effort_at_20_percent_recall,
    "initial_false_alarm": initial_false_alarm,
}


class Score(NamedTuple):
    loss: float
    accuracy_score: float = -1.0
    balanced_accuracy_score: float = -1.0
    roc_auc_score: float = -1.0
    matthews_corrcoef: float = -1.0
    recall_score: float = -1.0
    recall_at_20_percent_loc: float = -1.0
    effort_at_20_percent_recall: float = -1.0
    initial_false_alarm: float = -1.0

    @staticmethod
    @cache
    def get_scores_names():
        return set(score for score in Score._fields if score != "loss")

    def __repr__(self):
        """
        Make a representation of the score
        with each score in a new line
        """
        values = ",\n".join(
            f"    {score_name}={score_value}"
            for score_name, score_value in self._asdict().items()
        )

        return f"Score(\n{values}\n)"


class Scorer:
    @staticmethod
    def compute_score(
        metric_names: Iterable[str],
        labels: np.ndarray,
        true_prob: np.ndarray,
        loss=-1.0,
    ):
        threshold = Scorer.get_threshold_from_roc_curve(labels, true_prob)
        print("threshold:", threshold)
        predictions = true_prob > threshold

        scores = {"loss": loss}
        for score_name in metric_names:
            assert score_name in Score.get_scores_names(), score_name
            if hasattr(metrics, score_name):
                scorer = getattr(metrics, score_name)
            else:
                scorer = custom_metrics[score_name]
            args = {"y_true": labels}
            var_names = scorer.__code__.co_varnames

            if var_names[1] == "y_score":
                args["y_score"] = true_prob
            elif var_names[1] == "y_pred":
                args["y_pred"] = predictions
            else:
                num_metric_params = scorer.__code__.co_argcount
                raise ValueError(
                    f'{score_name}({",".join(var_names[:num_metric_params])})'
                )
            score = scorer(**args)
            scores[score_name] = score

        return Score(**scores)

    @staticmethod
    def get_threshold_from_roc_curve(labels, true_prob):
        """
        returns the threshold that maximizes balanced accuracy score
        """
        fpr, tpr, thresholds = metrics.roc_curve(labels, true_prob)
        tpr_fpr = tpr - fpr
        return thresholds[np.argmax(tpr_fpr)]
