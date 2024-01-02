from typing import Any
from argparse import Namespace
from colorama import Fore, Style

def dict_to_namespace(d: dict) -> Namespace:
    """Converts a dictionary to a namespace recursively."""
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)

def index_label_pred_to_lists(triples: list[dict[str, Any]]) -> tuple[list[int], list[str], list[str]]:
    """Convert (index, label, pred) triples to lists of indices, labels, and preds."""
    indices = []
    labels = []
    preds = []
    for triple in triples:
        indices.append(triple["index"])
        labels.append(triple["label"])
        preds.append(triple["pred"])
    return indices, labels, preds

def display_dialogue(dial: list[dict[str, str]]) -> None:
    """Display the dialogue in a human-readable format."""
    for utter in dial:
        if utter["role"] == "patient":
            print(f"Patient: {utter['utterance']}")
        elif utter["role"] == "doctor":
            print(f"{Fore.BLUE + Style.BRIGHT}Doctor: {utter['utterance']}{Style.RESET_ALL}")
        else:
            raise ValueError(f"Unknown role: {utter['role']}")

# manual testing
if __name__ == "__main__":
    d = {"a": {"c": 2, "d": {"e": 3}}, "b": 3}
    n = dict_to_namespace(d)
    print(n)
    print(n.a.c)
    print(n.a.d.e)
