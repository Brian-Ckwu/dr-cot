import json
from enum import Enum
from typing import Any
from pathlib import Path

from shot import Shot
from context import Context
from dialogue import Dialogue
from model import Model, APIModel

class Role(Enum):
    """The role of the bot in the dialogue."""

    PATIENT = "patient"
    DOCTOR = "doctor"

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

class DoctorBot(Bot):
    """The chat bot playing the doctor role."""

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
        self.role = Role.DOCTOR

# manual unit tests
if __name__ == "__main__":
    # test PatientBot.get_chatcompletion_prompt() and DoctorBot.get_chatcompletion_prompt()
    
    patient_prompt = json.loads(Path("../../prompts/patient/debug.json").read_bytes())
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
        model=APIModel()
    )
    chatcompletion_prompt = patient_bot.get_chatcompletion_prompt()
    print(json.dumps(chatcompletion_prompt, indent=4))

    doctor_prompt = json.loads(Path("../../prompts/doctor/debug.json").read_bytes())
    doctor_bot = DoctorBot(
        prefix_instruction=doctor_prompt["prefix_instruction"],
        shots=[
            Shot(
                context=Context(raw_text=shot["context"]),
                dialogue=Dialogue(data=shot["dialogue"])
            ) for shot in doctor_prompt["shots"]
        ],
        context=Context(raw_text=doctor_prompt["context"]),
        dialogue=Dialogue(data=doctor_prompt["dialogue"]),
        suffix_instruction=doctor_prompt["suffix_instruction"],
        model=APIModel()
    )
    chatcompletion_prompt = doctor_bot.get_chatcompletion_prompt()
    print(json.dumps(chatcompletion_prompt, indent=4))