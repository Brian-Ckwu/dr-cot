import json
from pathlib import Path
from argparse import ArgumentParser, Namespace
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from src import *

ALL_DXS = {
    'Anemia': 'ANEM',
    'Panic attack': 'PANC',
    'Influenza': 'FLU',
    'Boerhaave': 'BOER',
    'Bronchospasm / acute asthma exacerbation': 'BRSP',
    'Allergic sinusitis': 'ALSI',
    'Acute otitis media': 'AOM',
    'Pulmonary embolism': 'PEMB',
    'Viral pharyngitis': 'VIPH',
    'Myasthenia gravis': 'MGRA',
    'Bronchiectasis': 'BRONE',
    'SLE': 'SLE',
    'Bronchitis': 'BRONC',
    'Pneumonia': 'PNEU',
    'Inguinal hernia': 'INGH',
    'Acute dystonic reactions': 'ADYS',
    'Acute rhinosinusitis': 'ARSI',
    'Pericarditis': 'PERI',
    'Atrial fibrillation': 'AFIB',
    'Anaphylaxis': 'ANPH',
    'HIV (initial infection)': 'HIVI',
    'URTI': 'URTI',
    'Chronic rhinosinusitis': 'CRRS',
    'Cluster headache': 'CLHD',
    'Stable angina': 'STAN',
    'Spontaneous pneumothorax': 'SPPT',
    'Acute laryngitis': 'ACUL',
    'Pulmonary neoplasm': 'PNEO',
    'Myocarditis': 'MYOC',
    'Acute pulmonary edema': 'APE',
    'Unstable angina': 'UANG',
    'Scombroid food poisoning': 'SCOM',
    'PSVT': 'PSVT',
    'Acute COPD exacerbation / infection': 'COPD',
    'Localized edema': 'LOCE',
    'Guillain-Barré syndrome': 'GBSY',
    'Possible NSTEMI / STEMI': 'NSTE',
    'Pancreatic neoplasm': 'PANC',
    'Larygospasm': 'LARY',
    'Sarcoidosis': 'SARC',
    'Spontaneous rib fracture': 'SRF',
    'GERD': 'GERD',
    'Chagas': 'CHAG',
    'Croup': 'CROP',
    'Epiglottitis': 'EPIG',
    'Tuberculosis': 'TB',
    'Whooping cough': 'WHOOP',
    'Bronchiolitis': 'BRIO',
    'Ebola': 'EBLA'
}

def extract_label_set(labels: list[str]) -> list[str]:
    ls = set()
    for label in labels:
        if label in ALL_DXS:
            ls.add(label)
    return list(ls)

def display_confusion_matrix(
    indices: list[int],
    labels: list[str],
    preds: list[str],
    label_set: list[str],
    abbrs: list[str] = None
):
    """Display the confusion matrix for the given experiment directory."""
    label_set = sorted(label_set)
    metrics = Metrics(indices, labels, preds)
    cm = metrics.confusion_matrix(label_set)
    if abbrs is None:
        abbrs = [ALL_DXS[label] for label in label_set] + ["Other"]
    return ConfusionMatrixDisplay(cm, display_labels=abbrs)

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=Path,
        required=True,
        help="Path to the evaluation json file."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = setup_args()
    results = json.loads(args.json_path.read_bytes())
    acc = results["accuracy"]
    indices, labels, preds = index_label_pred_to_lists(results["label_pred_pairs"])
    label_set = extract_label_set(labels)
    print(f"Accuracy: {acc * 100:.2f}%")
    cmd = display_confusion_matrix(indices, labels, preds, label_set)
    cmd.plot()
    plt.show()
