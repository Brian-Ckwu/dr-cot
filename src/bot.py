import json
import yaml
from enum import Enum
from typing import Any
from pathlib import Path

from .shot import Shot
from .context import Context
from .dialogue import Dialogue, Role
from .model import Model, OpenAIModel

class PromptFormat(Enum):
    RAW_TEXT = "raw_text"
    JSON = "json"

class PromptMode(Enum):
    STANDARD = "standard"
    DRCoT = "drcot"

class Bot(object):
    """The chat bot playing a given role."""

    def __init__(
        self,
        prefix_instruction: str,
        shots: list[Shot],
        context: Context,
        dialogue: Dialogue,
        suffix_instruction: str,
        model: Model
    ):
        self.prefix_instruction = prefix_instruction
        self.shots = shots
        self.context = context
        self.dialogue = dialogue
        self.suffix_instruction = suffix_instruction
        self.model = model
        self.role = None

    def get_completion_prompt(self) -> str:
        """Get the completion prompt for the bot."""
        raise NotImplementedError

    def get_chatcompletion_prompt(self) -> list[dict[str, str]]:
        """Get the chatcompletion prompt for the bot."""
        if self.role is None:
            raise ValueError("Bot role is None.")
        msgs = []
        for shot in self.shots:
            # system message: prefix_instruction + shot_context_text
            msgs.append({
                "role": "system",
                "content": self.prefix_instruction + '\n' + shot.context.text()
            })
            # dialogue
            for d in shot.dialogue.data:
                msgs.append({
                    "role": "system",
                    "name": "example_" + d["role"], # help clarify that this is an example
                    "content": d["utterance"]
                })
        # current system message: prefix_instruction + context_text
        msgs.append({
            "role": "system",
            "content": self.prefix_instruction + '\n' + self.context.text()
        })
        # current diaglogue
        for d in self.dialogue.data:
            msgs.append({
                "role": "assistant" if d["role"] == self.role.value else "user",
                "name": d["role"],
                "content": d["utterance"]
            })
        # suffix_instruction if it exists
        if self.suffix_instruction:
            msgs.append({
                "role": "system",
                "content": self.suffix_instruction
            })
        return msgs
    
    def get_prompt(self) -> Any:
        if self.model.config["model"] in self.model.chatcompletion_models:
            prompt = self.get_chatcompletion_prompt()
        else:
            prompt = self.get_completion_prompt()
        return prompt
    
    def respond(self, utterance: str) -> str:
        """Respond to the counterpart chatbot's utterance."""
        self.dialogue.add_utterance(self.opposite_role, utterance)

        if isinstance(self.model, OpenAIModel):
            prompt = self.get_prompt()
            # print(json.dumps(prompt, indent=4)) # XXX
            response = self.model.generate(prompt)
        else:
            raise NotImplementedError

        # TODO: May need additional parsing here to separate inner monologue from actual response
        self.dialogue.add_utterance(self.role, response)
        return response
    
    def reset(self, context: Context, dialogue: Dialogue = None) -> None:
        self.set_context(context)
        self.clear_dialogue()
        if dialogue is not None:
            self.dialogue = dialogue
    
    def set_context(self, context: Context) -> None:
        """Set the context for the bot."""
        self.context = context
    
    def clear_dialogue(self) -> None:
        """Clear the dialogue for the bot."""
        self.dialogue = Dialogue([]) # [] is necessary to create a new dialogue object

class PatientBot(Bot):
    """The chat bot playing the patient role."""
    context_delimiter = "```" # currently triple backticks

    def __init__(
        self,
        prefix_instruction: str,
        shots: list[Shot],
        context: Context,
        dialogue: Dialogue,
        suffix_instruction: str,
        model: Model
    ):
        super().__init__(
            prefix_instruction,
            shots,
            context,
            dialogue,
            suffix_instruction,
            model
        )
        self.role = Role.PATIENT
        self.opposite_role = Role.DOCTOR
    
    def inform_initial_evidence(self, utterance: str) -> str:
        """Inform the initial evidence to the doctor."""
        self.dialogue.add_utterance(self.opposite_role, utterance)
        response = self.context.initial_evidence
        self.dialogue.add_utterance(self.role, response)
        return response
    
    @property
    def state(self) -> dict[str, Any]:
        """Get the state of the bot."""
        return {
            "prefix_instruction": self.prefix_instruction,
            "shots": [
                {
                    "context": shot.context.text(),
                    "dialogue": shot.dialogue.data
                } for shot in self.shots
            ],
            "context": self.context.text(),
            "dialogue": self.dialogue.data,
            "suffix_instruction": self.suffix_instruction,
            "model": self.model.config,
            "prompt": self.get_prompt()
        }

class DoctorBot(Bot):
    """The chat bot playing the doctor role."""
    greeting_msg = json.dumps({
        "action": "greeting",
        "question": "How may I help you today?"
    })
    ask_basic_info_msg = json.dumps({
        "action": "ask_basic_info",
        "question": "What's your sex and age?"
    })

    def __init__(
        self,
        prefix_instruction: str,
        shots: list[Shot],
        context: Context,
        dialogue: Dialogue,
        suffix_instruction: str,
        suffix_instructions: dict[str, str], # the doctor has different suffix instructions fr "ask_finding" and "make_diagnosis"
        model: Model,
        max_ddx: int = int(1e8),
        prompt_format: str = PromptFormat.RAW_TEXT.value,
    ):
        super().__init__(
            prefix_instruction,
            shots,
            context,
            dialogue,
            suffix_instruction,
            model
        )
        self.role = Role.DOCTOR
        self.opposite_role = Role.PATIENT
        self.suffix_instructions = suffix_instructions
        self.max_ddx = max_ddx
        self.set_max_ddx()
        if prompt_format not in [p.value for p in PromptFormat]:
            raise ValueError(f"Invalid prompt format: {prompt_format}")
        self.prompt_format = prompt_format
    
    def set_max_ddx(self) -> None:
        """Set the maximum number of differential diagnoses."""
        for shot in self.shots:
            shot.dialogue.reverse_parse_dialogue()
            for turn in shot.dialogue.data:
                if turn["role"] == self.role.value and isinstance(turn["utterance"], dict) and "ranked differential diagnosis" in turn["utterance"]:
                    turn["utterance"]["ranked differential diagnosis"] = turn["utterance"]["ranked differential diagnosis"][:self.max_ddx]
            shot.dialogue.parse_dialogue()
    
    def set_suffix_instruction(self, suffix_instruction: str) -> None:
        """Set the suffix instruction for the bot."""
        self.suffix_instruction = suffix_instruction
    
    def greeting(self, utterance: str = "") -> str:
        if utterance:
            self.dialogue.add_utterance(self.opposite_role, utterance)
        self.dialogue.add_utterance(self.role, self.greeting_msg)
        d = json.loads(self.greeting_msg)
        return d["question"]
    
    def ask_basic_info(self, utterance: str = "") -> str:
        if utterance:
            self.dialogue.add_utterance(self.opposite_role, utterance)
        else:
            raise ValueError("Utterance is empty.")
        self.dialogue.add_utterance(self.role, self.ask_basic_info_msg)
        d = json.loads(self.ask_basic_info_msg)
        return d["question"]
    
    def ask_finding(self, utterance: str) -> str:
        self.set_suffix_instruction(self.suffix_instructions["ask_finding"])
        response = self.respond(utterance)
        try:
            d = json.loads(response)
        except:
            print(f"===== Error response =====\n{response}\n")
            raise ValueError("Response is not a valid JSON string.")
        return d["question"]

    def make_diagnosis(self, utterance: str) -> str:
        self.set_suffix_instruction(self.suffix_instructions["make_diagnosis"])
        response = self.respond(utterance)
        try:
            d = json.loads(response)
        except:
            raise ValueError("Response is not a valid JSON string.")
        return d["most likely diagnosis"]

    @property
    def state(self) -> dict[str, Any]:
        """Get the state of the bot."""
        return {
            "prefix_instruction": self.prefix_instruction,
            "shots": [
                {
                    "context": shot.context.text(),
                    "dialogue": shot.dialogue.data
                } for shot in self.shots
            ],
            "context": self.context.text(),
            "dialogue": self.dialogue.data,
            "suffix_instructions": self.suffix_instructions,
            "model": self.model.config,
            "prompt": self.get_prompt()
        }

# manual unit tests
if __name__ == "__main__":
    from data import DDxDataset
    # Load dataset
    csv_path = "../../ddxplus/release_test_patients.csv"
    pathology_info_path = "../../ddxplus/release_conditions.json"
    evidences_info_path = "../../ddxplus/our_evidences_to_qa_v2.json"

    dataset = DDxDataset(csv_path, pathology_info_path, evidences_info_path)
    indices = [98595] #, 123464, 86477, 9209, 98151]
    pats = dataset.df.iloc[indices]
    print("Dataset loaded.")

    # test PatientBot.get_chatcompletion_prompt() and DoctorBot.get_chatcompletion_prompt()
    with open("../../experiments/configs/debug.yml") as f:
        args = yaml.safe_load(f)
    
    patient_prompt = json.loads(Path("../../prompts/patient/standard.json").read_bytes())
    
    patient_bot = PatientBot(
        prefix_instruction=patient_prompt["prefix_instruction"],
        shots=[
            Shot(
                context=Context(raw_text=shot["context"]),
                dialogue=Dialogue(data=shot["dialogue"])
            ) for shot in patient_prompt["shots"]
        ],
        context=Context(raw_text=patient_prompt["context"]),
        dialogue=Dialogue(data=patient_prompt["dialogue"]),
        suffix_instruction=patient_prompt["suffix_instruction"],
        model=OpenAIModel(config=args["patient"]["model_config"])
    )
    
    question = "What's your sex and age?"
    
    from context import PatientContext
    for _, pat in pats.iterrows():
        print(f"Patient: {pat.AGE} yo {pat.SEX}")
        patient_bot.reset(
            context=PatientContext(
                sex=pat.SEX,
                age=pat.AGE,
                initial_evidence=pat.INITIAL_EVIDENCE,
                evidences=pat.EVIDENCES
            )
        )
        response = patient_bot.respond(question)
        print(response)
        print(patient_bot.dialogue.data)
        print(patient_bot.get_chatcompletion_prompt())

    # doctor_prompt = json.loads(Path("../../prompts/doctor/debug.json").read_bytes())
    # doctor_bot = DoctorBot(
    #     prefix_instruction=doctor_prompt["prefix_instruction"],
    #     shots=[
    #         Shot(
    #             context=Context(raw_text=shot["context"]),
    #             dialogue=Dialogue(data=shot["dialogue"])
    #         ) for shot in doctor_prompt["shots"]
    #     ],
    #     context=Context(raw_text=doctor_prompt["context"]),
    #     dialogue=Dialogue(data=doctor_prompt["dialogue"]),
    #     suffix_instruction=doctor_prompt["suffix_instruction"],
    #     model=OpenAIModel(config=args["doctor"]["model_config"])
    # )
    
    # answer = "No. I don't have a fever."
    # response = doctor_bot.respond(answer)
    # print(response)
    # print(doctor_bot.dialogue.data)
