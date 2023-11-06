import json
from pathlib import Path

class Metrics:

    def __init__(self, indices: list[int], labels: list[str], preds: list[str]) -> None:
        if not (len(indices) == len(labels) == len(preds)):
            raise ValueError(f"indices, labels, and preds must have the same length, but got {len(indices)}, {len(labels)}, and {len(preds)}.")
        self.indices = indices
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
            "label_pred_pairs": [{"index": index, "label": label, "pred": pred} for index, label, pred in zip(self.indices, self.labels, self.preds)]
        }
        Path(save_path).write_text(json.dumps(d, indent=4))
