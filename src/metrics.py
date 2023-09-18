import json
from pathlib import Path

class Metrics:

    def __init__(self, labels: list[str], preds: list[str]) -> None:
        if len(labels) != len(preds):
            raise ValueError(f"labels and preds must have the same length, but got {len(labels)} and {len(preds)}.")
        self.labels = labels
        self.preds = preds

    @property
    def accuracy(self) -> float:
        ncorrect = 0
        for i in range(len(self.labels)):
            if self.labels[i] == self.preds[i]:
                ncorrect += 1
        return ncorrect / len(self.labels)

    def save_results(self, save_path: str) -> None:
        d = {
            "accuracy": self.accuracy,
            "label_pred_pairs": [{"label": label, "pred": pred} for label, pred in zip(self.labels, self.preds)]
        }
        Path(save_path).write_text(json.dumps(d, indent=4))
