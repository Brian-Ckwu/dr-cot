import re
import json
import yaml
from enum import Enum
from typing import Any, Union
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

class Action(Enum):
    GREETING = "greeting"
    ASK_FINDING = "ask_finding"
    MAKE_DIAGNOSIS = "make_diagnosis"

class ReasoningStep(Enum):
    POS_FINDINGS = "positive clinical findings"
    NEG_FINDINGS = "negative clinical findings"
    RANKED_DDX = "ranked differential diagnosis"
    ASK_FINDING = "the clinical finding to ask about"
    QUESTION = "question"
    DX_RATIONALE = "rationale"
    FINAL_DIAGNOSIS = "most likely diagnosis"

class Bot(object):
    """The chat bot playing a given role."""
    history_taking_msg = "<History taking>"

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
        raise NotImplementedError
    
    def get_prompt(self) -> Any:
        if self.model.config["model"] in self.model.chatcompletion_models:
            prompt = self.get_chatcompletion_prompt()
        else:
            prompt = self.get_completion_prompt()
        return prompt

    def get_role_string(self) -> str:
        return self.role.value[0].upper() + self.role.value[1:]

    def respond(self, utterance: str) -> str:
        """Respond to the counterpart chatbot's utterance."""
        self.dialogue.add_utterance(self.opposite_role, utterance)

        prompt = self.get_prompt()
        response = self.model.generate(prompt).strip()

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

    def get_completion_prompt(self) -> str:
        """Get the completion prompt (a whole string) for the bot."""
        sents = []
        for shot in self.shots:
            sents.append(self.prefix_instruction)
            sents.append(shot.context.text())
            sents.append('')
            sents.append(self.history_taking_msg)
            for d in shot.dialogue.data:
                role_str = d["role"][0].upper() + d["role"][1:]
                sent = f"{role_str}: {d['utterance']}"
                sents.append(sent)
            sents.append('')
        sents.append(self.prefix_instruction)
        sents.append(self.context.text())
        sents.append('')
        sents.append(self.history_taking_msg)
        for d in self.dialogue.data:
            role_str = d["role"][0].upper() + d["role"][1:]
            sent = f"{role_str}: {d['utterance']}"
            sents.append(sent)
        suffix = ''
        if self.suffix_instruction:
            suffix = self.suffix_instruction
        return '\n'.join(sents) + f"\n{self.get_role_string()}: {suffix}"

    def get_chatcompletion_prompt(self) -> list[dict[str, str]]:
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
    greeting_msg = {
        "action": "greeting",
        "question": "How may I help you today?"
    }
    final_diagnosis_msg = "Based on your description, the most likely diagnosis is"
    ask_finding_prefix = "[Ask finding]"
    make_diagnosis_prefix = "[Make diagnosis]"

    def __init__(
        self,
        prefix_instruction: str,
        shots: list[Shot],
        context: Context,
        dialogue: Dialogue,
        suffix_instruction: str,
        suffix_instructions: dict[str, str], # the doctor has different suffix instructions fr "ask_finding" and "make_diagnosis"
        model: Model,
        max_ddx: int,
        prompt_mode: str,
        prompt_format: str,
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
        if prompt_mode not in [p.value for p in PromptMode]:
            raise ValueError(f"Invalid prompt mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        if prompt_format not in [p.value for p in PromptFormat]:
            raise ValueError(f"Invalid prompt format: {prompt_format}")
        self.prompt_format = prompt_format

    def parse_utterance(self, utterance: Union[str, dict[str, Any]]) -> str:
        """Parse the utterance according to prompt_mode and prompt_format."""
        if isinstance(utterance, str):
            return utterance
        if self.prompt_mode == PromptMode.STANDARD.value:
            if self.prompt_format == PromptFormat.JSON.value:
                d = {"action": utterance["action"]}
                if utterance["action"] == Action.MAKE_DIAGNOSIS.value:
                    d[ReasoningStep.FINAL_DIAGNOSIS.value] = utterance[ReasoningStep.FINAL_DIAGNOSIS.value]
                elif utterance["action"] in [Action.ASK_FINDING.value, Action.GREETING.value]:
                    d[ReasoningStep.QUESTION.value] = utterance[ReasoningStep.QUESTION.value]
                else:
                    raise ValueError(f"Invalid action: {utterance['action']}")
                return json.dumps(d)
            elif self.prompt_format == PromptFormat.RAW_TEXT.value:
                if utterance["action"] == Action.MAKE_DIAGNOSIS.value:
                    return f"{self.final_diagnosis_msg} {utterance[ReasoningStep.FINAL_DIAGNOSIS.value]}."
                elif utterance["action"] in [Action.ASK_FINDING.value, Action.GREETING.value]:
                    return utterance[ReasoningStep.QUESTION.value]
                else:
                    raise ValueError(f"Invalid action: {utterance['action']}")
            else:
                raise ValueError(f"Invalid prompt format: {self.prompt_format}")
        elif self.prompt_mode == PromptMode.DRCoT.value:
            if self.prompt_format == PromptFormat.JSON.value:
                raise NotImplementedError
            elif self.prompt_format == PromptFormat.RAW_TEXT.value:
                sents = []
                if utterance["action"] == Action.GREETING.value:
                    sents.append(utterance[ReasoningStep.QUESTION.value])
                elif utterance["action"] in [Action.ASK_FINDING.value, Action.MAKE_DIAGNOSIS.value]:
                    if utterance["action"] == Action.ASK_FINDING.value:
                        prefix = self.ask_finding_prefix
                    else:  # Action.MAKE_DIAGNOSIS.value
                        prefix = self.make_diagnosis_prefix
                    symptom_review = f"""Based on the {ReasoningStep.POS_FINDINGS.value} '{", ".join(utterance[ReasoningStep.POS_FINDINGS.value])}' and the {ReasoningStep.NEG_FINDINGS.value} '{", ".join(utterance[ReasoningStep.NEG_FINDINGS.value])}',"""
                    dd_formulation = f"""the {ReasoningStep.RANKED_DDX.value} is '{", ".join(utterance[ReasoningStep.RANKED_DDX.value])}'."""
                    sents += [prefix, symptom_review, dd_formulation]
                    if utterance["action"] == Action.ASK_FINDING.value:
                        next_inquiry = f"""To narrow down the {ReasoningStep.RANKED_DDX.value}, the {ReasoningStep.ASK_FINDING.value} is '{utterance[ReasoningStep.ASK_FINDING.value]}'."""
                        question = f"""[{ReasoningStep.QUESTION.value}] {utterance[ReasoningStep.QUESTION.value]}"""
                        sents += [next_inquiry, question]
                    else:  # Action.MAKE_DIAGNOSIS.value
                        dx_rationale = utterance[ReasoningStep.DX_RATIONALE.value]
                        final_dx = f"""[{ReasoningStep.FINAL_DIAGNOSIS.value}] {utterance[ReasoningStep.FINAL_DIAGNOSIS.value]}"""
                        sents += [dx_rationale, final_dx]
                return ' '.join(sents)
            else:
                raise ValueError(f"Invalid prompt format: {self.prompt_format}")
        else:
            raise ValueError(f"Invalid prompt mode: {self.prompt_mode}")

    def get_completion_prompt(self) -> str:
        """Get the completion prompt (a whole string) for the bot."""
        instruction = self.prefix_instruction + '\n' + self.context.text() + '\n'
        sents = [instruction]
        for shot in self.shots:
            sents.append(self.history_taking_msg)
            for d in shot.dialogue.data:
                role_str = d["role"][0].upper() + d["role"][1:]
                sent = f"{role_str}: {self.parse_utterance(d['utterance'])}"
                sents.append(sent)
            sents.append('')
        sents.append(self.history_taking_msg)
        for d in self.dialogue.data:
            role_str = d["role"][0].upper() + d["role"][1:]
            sent = f"{role_str}: {self.parse_utterance(d['utterance'])}"
            sents.append(sent)
        suffix = ''
        if self.suffix_instruction:
            suffix = self.suffix_instruction
        return '\n'.join(sents) + f"\n{self.get_role_string()}: {suffix}"

    def get_chatcompletion_prompt(self) -> list[dict[str, str]]:
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
                    "content": d["utterance"] if d["role"] == Role.PATIENT.value else self.parse_utterance(d["utterance"])
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
    
    def set_max_ddx(self) -> None:
        """Set the maximum number of differential diagnoses."""
        for shot in self.shots:
            for turn in shot.dialogue.data:
                if turn["role"] == self.role.value and isinstance(turn["utterance"], dict) and "ranked differential diagnosis" in turn["utterance"]:
                    turn["utterance"]["ranked differential diagnosis"] = turn["utterance"]["ranked differential diagnosis"][:self.max_ddx]
    
    def set_suffix_instruction(self, suffix_instruction: str) -> None:
        """Set the suffix instruction for the bot."""
        self.suffix_instruction = suffix_instruction
    
    def greeting(self, utterance: str = "") -> str:
        if utterance:
            self.dialogue.add_utterance(self.opposite_role, utterance)
        if self.prompt_format == PromptFormat.JSON.value:
            greeting_msg = json.dumps(self.greeting_msg)
        elif self.prompt_format == PromptFormat.RAW_TEXT.value:
            greeting_msg = self.greeting_msg["question"]
        else:
            raise ValueError(f"Invalid prompt format: {self.prompt_format}")
        self.dialogue.add_utterance(role=self.role, utterance=greeting_msg)
        return self.greeting_msg["question"]
    
    def ask_basic_info(self, utterance: str = "") -> str:
        if utterance:
            self.dialogue.add_utterance(self.opposite_role, utterance)
        else:
            raise ValueError("Utterance is empty.")
        self.dialogue.add_utterance(self.role, self.ask_basic_info_msg)
        d = json.loads(self.ask_basic_info_msg)
        return d["question"]

    def parse_response(self, response: str, key: str) -> str:
        if self.prompt_format == PromptFormat.JSON.value:
            try:
                d = json.loads(response)
            except:
                print(f"===== Error response =====\n{response}\n")
                raise ValueError("Response is not a valid JSON string.")
            return d[key]
        elif self.prompt_format == PromptFormat.RAW_TEXT.value:
            if self.prompt_mode == PromptMode.STANDARD.value:
                return response
            elif self.prompt_mode == PromptMode.DRCoT.value:
                found = re.findall(f"\\[{key}\\] (.*)", response)
                if len(found) > 0:
                    response = found[0]
                elif key == ReasoningStep.QUESTION.value:  # len(found) == 0
                    dx = re.findall(f"\\[{ReasoningStep.FINAL_DIAGNOSIS.value}\\] (.*)", response)  # early termination of the dialogue (the doctor bot make a diagnosis)
                    if len(dx) > 0:
                        response = dx[0]
                    else:
                        response = None
                elif key == ReasoningStep.FINAL_DIAGNOSIS.value:  # len(found) == 0
                    response = None
                else:
                    raise ValueError(f"Invalid key: {key}")
                return response
        else:
            raise ValueError(f"Invalid prompt format: {self.prompt_format}")

    def respond(self, utterance: str) -> str:
        """Respond to the PatientBot's utterance."""
        self.dialogue.add_utterance(self.opposite_role, utterance)

        prompt = self.get_prompt()
        response = self.model.generate(prompt).strip()

        self.dialogue.add_utterance(self.role, self.suffix_instruction + ' ' + response)
        return response

    def ask_finding(self, utterance: str) -> str:
        self.set_suffix_instruction(f"{self.ask_finding_prefix}")
        response = self.respond(utterance)
        return self.parse_response(response, key=ReasoningStep.QUESTION.value)

    def make_diagnosis(self, utterance: str) -> str:
        self.set_suffix_instruction(f"{self.make_diagnosis_prefix}")
        response = self.respond(utterance)
        return self.parse_response(response, key=ReasoningStep.FINAL_DIAGNOSIS.value)

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
