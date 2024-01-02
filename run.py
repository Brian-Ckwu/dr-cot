import os
import re
import time
import json
import yaml
import shutil
import pandas as pd
from pathlib import Path
from colorama import Fore, Style
from argparse import ArgumentParser, Namespace

from src import model
from src import *

class Experiment(object):

    def __init__(self, config: Namespace, debug: bool = False, api_interval: float = 1.0) -> None:
        self.config = config
        self.debug = debug
        self.api_interval = api_interval
        print(f"Loading dataset...")
        dataset = DDxDataset(**vars(config.data.dataset))
        self.pats = dataset.sample_patients(
            ie=config.data.initial_evidence,
            n=config.data.sample_size,
            seed=config.seed,
            ddxs=config.data.possible_diagnoses
        )
        print(f"Sampled {len(self.pats)} patients of initial evidence {config.data.initial_evidence}.")
        self.patient_bot = self.initialize_patient()
        self.doctor_bot = self.initialize_doctor(possible_diagnoses=config.data.possible_diagnoses)
        self.log_bots_info()

    def initialize_patient(self) -> PatientBot:
        patient_config = json.loads(Path(self.config.patient.config_path).read_bytes())
        return PatientBot(
            prefix_instruction=patient_config["prefix_instruction"],
            shots=[
                Shot(
                    context=Context(raw_text=shot["context"]),
                    dialogue=Dialogue(data=shot["dialogue"])
                ) for shot in patient_config["shots"]
            ],
            context=Context(raw_text=patient_config["context"]),
            dialogue=Dialogue(data=[]),
            suffix_instruction=patient_config["suffix_instruction"],
            model=getattr(model, config.patient.model_type)(config=config.patient.model_config)
        )
    
    def initialize_doctor(self, possible_diagnoses: list[str]) -> DoctorBot:
        return DoctorBot(
            prefix_instruction=Path(os.path.join("./prompts/doctor/prefix_instructions", f"{self.config.doctor.prompt_mode}.txt")).read_text(),
            shots=[
                Shot(
                    context=DoctorContext(possible_diagnoses),
                    dialogue=Dialogue(data=shot["dialogue"])
                ) for shot in [json.loads(Path(path).read_bytes()) for path in self.config.doctor.shots_paths]
            ],
            context=DoctorContext(possible_diagnoses),
            dialogue=Dialogue(data=[]),
            suffix_instruction="",
            suffix_instructions=json.loads((Path("./prompts/doctor/suffix_instructions") / f"{self.config.doctor.prompt_mode}.json").read_bytes()),
            model=getattr(model, config.doctor.model_type)(config=self.config.doctor.model_config),
            max_ddx=self.config.doctor.max_ddx,
            prompt_mode=self.config.doctor.prompt_mode,
            prompt_format=self.config.doctor.prompt_format
        )

    def get_new_patient_context(self, pat: pd.Series) -> PatientContext:
        return PatientContext(
            sex=pat.SEX,
            age=pat.AGE,
            initial_evidence=pat.INITIAL_EVIDENCE,
            evidences=pat.EVIDENCES
        )

    def log_bots_info(self) -> None:
        (self.config.log_path / "patient_state.json").write_text(json.dumps(self.patient_bot.state, indent=4))
        (self.config.log_path / "doctor_state.json").write_text(json.dumps(self.doctor_bot.state, indent=4))
        self.config.patient_log_path = self.config.log_path / "patient_dialogues"
        self.config.profile_log_path = self.config.log_path / "patient_profiles"
        self.config.doctor_log_path = self.config.log_path / "doctor_dialogues"
        self.config.patient_log_path.mkdir(parents=True, exist_ok=True)
        self.config.profile_log_path.mkdir(parents=True, exist_ok=True)
        self.config.doctor_log_path.mkdir(parents=True, exist_ok=True)

    def save_dialogues(self, index: int) -> None:
        self.patient_bot.dialogue.save_dialogue(save_path=self.config.patient_log_path / f"{index}.json", is_json=False)
        self.doctor_bot.dialogue.save_dialogue(save_path=self.config.doctor_log_path / f"{index}.json", is_json=(self.config.doctor.prompt_format == "json"))

    def save_patient_profile(self, index: int) -> None:
        (self.config.profile_log_path / f"{index}.txt").write_text(self.patient_bot.context.text())

    def conduct_history_taking(self, doctor_bot: DoctorBot, patient_bot: PatientBot, dialogue_index: int) -> str:
        """Conduct history taking with the given doctor and patient bots."""
        q = doctor_bot.greeting()
        a = patient_bot.inform_initial_evidence(utterance=q)
        for i in range(self.config.doctor.ask_turns):
            q = doctor_bot.ask_finding(utterance=a)
            if (q == doctor_bot.NONE_RESPONSE) or (q[-1] != '?'):  # no more questions
                self.save_dialogues(index=dialogue_index)
                dx = q
                return dx
            a = patient_bot.respond(utterance=q)
            if (a == patient_bot.NONE_RESPONSE):
                self.save_dialogues(index=dialogue_index)
                return patient_bot.NONE_RESPONSE
            if self.debug:
                print(f"----- Turn {i + 1} -----")
                print(f"Doctor: {q}")
                print(f"Patient: {a}")
                self.save_dialogues(index=dialogue_index)
            time.sleep(self.api_interval)
        dx = doctor_bot.make_diagnosis(utterance=a)
        self.save_dialogues(index=dialogue_index)
        return dx

    def run(self) -> None:
        """Run the experiment with the given configuration."""
        for i, pat in self.pats.iterrows():
            if self.debug:
                print(f"\n===== Patient Index {i} =====\n")
            patient_context = self.get_new_patient_context(pat)
            self.patient_bot.reset(context=patient_context)
            self.doctor_bot.clear_dialogue()
            self.save_patient_profile(index=i)
            dx = self.conduct_history_taking(self.doctor_bot, self.patient_bot, dialogue_index=i)
            print(f"Ground truth: {pat.PATHOLOGY}; Prediction: {dx}")

    def extract_dx(self, utterance: str) -> str:
        """Extract the diagnosis from the given utterance."""
        if self.config.doctor.prompt_mode == PromptMode.STANDARD.value:
            found = re.findall(f"{self.doctor_bot.final_diagnosis_msg} (.*)", utterance)
            if len(found) == 1:
                utterance = found[0].strip().rstrip('.')
        elif self.config.doctor.prompt_mode == PromptMode.DRCoT.value:
            found = re.findall(f"\\[{ReasoningStep.FINAL_DIAGNOSIS.value}\\] (.*)", utterance)
            if len(found) == 1:
                utterance = found[0].strip().rstrip('.')
        return utterance

    def evaluate(self) -> None:
        """Evaluate the experiment with the given configuration."""
        ncorrect = 0
        indices = []
        labels = []
        preds = []
        for i, pat in self.pats.iterrows():
            filepath = self.config.doctor_log_path / f"{i}.json"
            if not filepath.exists():
                raise ValueError(f"File {filepath} does not exist.")
            final_utter = json.loads(filepath.read_bytes())[-1]
            if not final_utter["role"] == Role.DOCTOR.value:
                raise ValueError(f"Final utterance is not from doctor.")
            label = pat.PATHOLOGY
            pred = self.extract_dx(final_utter["utterance"])
            indices.append(i)
            labels.append(label)
            preds.append(pred)
            ncorrect += int(pred == label)
            if self.debug:
                print(f"{str(i).zfill(6)} -> Ground Truth: {Fore.RED + label + Style.RESET_ALL} / Prediction: {Fore.BLUE + pred + Style.RESET_ALL}{f' {Fore.GREEN}✔{Style.RESET_ALL}' if (pred == label) else ''}")
        metrics = Metrics(indices, labels, preds)
        metrics.save_results(save_path=self.config.log_path / "eval_results.json")
        print(f"\nAccuracy: {metrics.accuracy * 100:.2f}% (Correct: {ncorrect} / Predicted: {len(self.pats)})")

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--api_interval", type=float, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--evaluate", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    # copy config file to output directory
    config.log_path = Path(config.log_path) / config.doctor.model_config.model.split('/')[-1] / config.doctor.prompt_mode / config.data.initial_evidence / config.name
    config.log_path.mkdir(parents=True, exist_ok=True)
    experiment = Experiment(config, debug=args.debug, api_interval=args.api_interval)
    if args.evaluate:
        experiment.evaluate()
    else:
        shutil.copy(args.config_path, config.log_path / "config.yaml")
        experiment.run()
