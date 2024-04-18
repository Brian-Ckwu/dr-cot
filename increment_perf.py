"""
The script for extracting intermediate states and performance of a given diagnostic dialogue.
"""
import yaml
import json
import jsonlines
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace

from src.utils import index_label_pred_to_lists, parse_json_from_string
from src.model import Model, GoogleModel

DX_PROMPT = """\
Act as a doctor and determine the most likely diagnosis according to the medical dialogue history.

Here is the medical dialogue history between a doctor and a patient:
{dialogue}

Here are the possible diagnoses you can choose from (each separated by a newline character):
{diagnoses}

Provide one most likely diagnosis in the following JSON format: {{"diagnosis": ""}}"""

DX_PROMPT_DRCOT = """\
Act as a doctor and determine the most likely diagnosis according to the medical dialogue history.

Here is the medical dialogue history between a doctor and a patient:
{dialogue}

Here are the possible diagnoses you can choose from (each separated by a newline character):
{diagnoses}

According to the medical dialogue history, follow the step-by-step reasoning plan in JSON to make an accurate final diagnosis. Provide your output in the following JSON format: {{"positive_clinical_findings": [], "negative_clinical_findings": [], "rationale_for_the_final_diagnosis": "", "diagnosis": ""}}"""

def extract_dxs(llm: Model, dial: list[dict], dxs_set: list[str], prompt_template: str) -> list[str]:
    dxs_text = '\n'.join(dxs_set)
    dxs_list = list()
    assert len(dial) % 2 == 0
    dial_lines = list()
    for i in range(0, len(dial), 2):
        assert (dial[i]["role"] == "doctor") and (dial[i + 1]["role"] == "patient")
        dial_lines.append(f"{dial[i]['role']}: {dial[i]['utterance']}")
        dial_lines.append(f"{dial[i + 1]['role']}: {dial[i + 1]['utterance']}")
        dial_text = '\n'.join(dial_lines)
        prompt = prompt_template.format(dialogue=dial_text, diagnoses=dxs_text)
        res = llm.generate(prompt)
        parsed_obj = parse_json_from_string(res)
        dx = parsed_obj["diagnosis"]
        dxs_list.append(dx)
    return dxs_list

def extract_dxs_pipe(args: Namespace) -> None:
    output_filename = "dx_extraction.jsonl"
    prompt_template = DX_PROMPT
    if args.drcot:
        output_filename = "dx_extraction_drcot.jsonl"
        prompt_template = DX_PROMPT_DRCOT
    llm = GoogleModel(config={
        "model": "gemini-1.0-pro",
        "temperature": 0.0,
        "candidate_count": 1,
        "max_output_tokens": 256,
    })
    config = yaml.safe_load((args.exp_dir / "config.yaml").read_text())
    dxs_set = config["data"]["possible_diagnoses"]
    dial_dir = args.exp_dir / "patient_dialogues"
    eval_results = json.loads((args.exp_dir / "eval_results.json").read_bytes())
    output_file = args.exp_dir / output_filename
    indices = set()
    if output_file.exists():
        with jsonlines.open(output_file) as f:
            rows = list(f)
            indices = {row["index"] for row in rows}
    for triple in tqdm(eval_results["label_pred_pairs"], desc="Extracting diagnoses per turn:"):
        index = triple["index"]
        if index in indices:
            continue
        dial_path = dial_dir / f"{index}.json"
        dial = json.loads(dial_path.read_bytes())
        dxs_list = extract_dxs(llm, dial, dxs_set, prompt_template)
        with open(output_file, mode="a") as f:
            f.write(json.dumps({"index": index, "dxs_list": dxs_list}) + '\n')
        if args.debug:
            break

def calc_acc_pipe(args: Namespace) -> list[float]:
    dxs_list_filename = "dx_extraction.jsonl"
    output_filename = "accs_per_turn.json"
    if args.drcot:
        dxs_list_filename = "dx_extraction_drcot.jsonl"
        output_filename = "accs_per_turn_drcot.json"
    dxs_list_file = args.exp_dir / dxs_list_filename
    extract_dxs_pipe(args)  # Automatically run this
    # Load rows of dxs_list
    with jsonlines.open(dxs_list_file) as f:
        rows = list(f)
    # Load ground truth
    eval_results = json.loads((args.exp_dir / "eval_results.json").read_bytes())
    indices, labels, _ = index_label_pred_to_lists(triples=eval_results["label_pred_pairs"])
    assert len(indices) == len(labels)
    index2label = {indices[i]: labels[i] for i in range(len(indices))}
    correct_counts = [0] * len(rows[0]["dxs_list"])
    for row in rows:
        for i, dx in enumerate(row["dxs_list"]):
            if dx == index2label[row["index"]]:
                correct_counts[i] += 1
        if args.debug:
            break
    accs = [cnt / len(rows) for cnt in correct_counts]
    output_path = args.exp_dir / output_filename
    output_path.write_text(json.dumps(obj={"accs": accs}))
    print(f"Accuracies per turn: {accs}")
    return accs

def setup_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=Path,
        required=True,
        help="Path to the experiment folder"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["extract_dxs", "calc_acc"],
        required=True,
        help="Which pipeline to run"
    )
    parser.add_argument(
        "--drcot",
        action="store_true",
        help="Use DRCoT in each diagnosis prediction or not."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode or not"
    )
    return parser.parse_args()

PIPELINES = {
    "extract_dxs": extract_dxs_pipe,
    "calc_acc": calc_acc_pipe
}

if __name__ == "__main__":
    args = setup_args()
    PIPELINES[args.pipeline](args)
