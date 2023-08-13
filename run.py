import os
import json
import yaml
import shutil
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, Namespace

from src import *

class Experiment(object):

    def __init__(self, config: Namespace, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        print(f"Loading dataset...")
        dataset = DDxDataset(**vars(config.data))
        self.pats = dataset.sample_patients(
            ie=config.initial_evidence,
            n=config.sample_size,
            seed=config.seed
        )
        print(f"Sampled {len(self.pats)} patients of initial evidence {config.initial_evidence}.")
        self.patient_bot = self.initialize_patient()
        self.doctor_bot = self.initialize_doctor(possible_diagnoses=dataset.get_all_diagnoses())
    
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
            model=OpenAIModel(config=config.patient.model_config)
        )
    
    def initialize_doctor(self, possible_diagnoses: list[str]) -> DoctorBot:
        return DoctorBot(
            prefix_instruction=Path(os.path.join("./prompts/doctor/prefix_instructions", f"{config.doctor.prompt_mode}.txt")).read_text(),
            shots=[
                Shot(
                    context=DoctorContext(possible_diagnoses),
                    dialogue=Dialogue(data=shot["dialogue"])
                ) for shot in [json.loads(Path(path).read_bytes()) for path in config.doctor.shots_paths]
            ],
            context=DoctorContext(possible_diagnoses),
            dialogue=Dialogue(data=[]),
            suffix_instruction="",
            suffix_instructions=json.loads((Path("./prompts/doctor/suffix_instructions") / f"{config.doctor.prompt_mode}.json").read_bytes()),
            model=OpenAIModel(config=config.doctor.model_config)
        )

    def get_new_patient_context(self, pat: pd.Series) -> PatientContext:
        return PatientContext(
            sex=pat.SEX,
            age=pat.AGE,
            initial_evidence=pat.INITIAL_EVIDENCE,
            evidences=pat.EVIDENCES
        )
    
    def conduct_history_taking(self, doctor_bot: DoctorBot, patient_bot: PatientBot) -> str:
        """Conduct history taking with the given doctor and patient bots."""
        q = doctor_bot.greeting()
        a = patient_bot.inform_initial_evidence(utterance=q)
        for _ in range(self.config.doctor.ask_turns):
            q = doctor_bot.ask_finding(utterance=a)
            a = patient_bot.respond(utterance=q)
            if self.debug:
                print(f"Doctor: {q}")
                print(f"Patient: {a}")
        dx = doctor_bot.make_diagnosis(utterance=a)
        return dx

    def run(self) -> None:
        """Run the experiment with the given configuration."""
        for _, pat in self.pats.iterrows():
            patient_context = self.get_new_patient_context(pat)
            self.patient_bot.reset(context=patient_context)
            self.doctor_bot.clear_dialogue()
            dx = self.conduct_history_taking(self.doctor_bot, self.patient_bot)
            print(f"Ground truth: {pat.PATHOLOGY}; Prediction: {dx}")

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--config_path", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    # copy config file to output directory
    config.log_path = Path(config.log_path) / config.doctor.model_config.model / config.doctor.prompt_mode / config.initial_evidence / config.name
    config.log_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_path, config.log_path / "config.yaml")
    experiment = Experiment(config, debug=args.debug)
    experiment.run()
