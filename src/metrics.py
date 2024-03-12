import json
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

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
            if ''.join(self.labels[i].upper().split()) == ''.join(self.preds[i].upper().split()):
                ncorrect += 1
        return ncorrect / len(self.labels)

    def confusion_matrix(self, label_set: list[str]) -> np.ndarray:
        label_dict = dict()
        for label in label_set:
            label_dict[label] = len(label_dict)
        y_true = [label_dict.get(label, len(label_set)) for label in self.labels]
        y_pred = [label_dict.get(pred, len(label_set)) for pred in self.preds]  # len(label_set) -> "OTHER"
        return confusion_matrix(y_true, y_pred, labels=range(len(label_set) + 1))

    def get_indices(self, label: str, pred: str) -> list[int]:
        """Get the indices of the given label and pred."""
        return [index for index, label_, pred_ in zip(self.indices, self.labels, self.preds) if (label_ == label and pred_ == pred)]

    def save_results(self, save_path: str) -> None:
        d = {
            "accuracy": self.accuracy,
            "label_pred_pairs": [{"index": index, "label": label, "pred": pred} for index, label, pred in zip(self.indices, self.labels, self.preds)]
        }
        Path(save_path).write_text(json.dumps(d, indent=4))
