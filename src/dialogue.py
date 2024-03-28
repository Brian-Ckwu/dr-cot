import json
import copy
from enum import Enum
from pathlib import Path

class Role(Enum):
    """The role of the bot in the dialogue."""

    PATIENT = "patient"
    DOCTOR = "doctor"

class Dialogue(object):
    """The dialogue between the PatientBot and the DoctorBot.
    
    Format:
    [
        {
            "role": str, # Role.value
            "utterance": str
        },
        ...
    ]
    """

    def __init__(self, data: list[dict[str, str]] = []):
        self.data = data
        # XXX: self.parse_dialogue()
    
    def __len__(self) -> int:
        """Return the number of utterances in the dialogue."""
        return len(self.data)

    def text(self) -> str:
        sents = list()
        for turn in self.data:
            role_text = turn["role"][0].upper() + turn["role"][1:]
            sent = f"{role_text}: {turn['utterance']}"
            sents.append(sent)
        return "\n".join(sents)

    def add_utterance(
        self,
        role: Role,
        utterance: str
    ):
        """Add an utterance to the dialogue."""
        self.data.append({
            "role": role.value,
            "utterance": utterance
        })
    
    def parse_dialogue(self) -> None:
        """Convert utterances which are dicts to strings in-place."""
        for turn in self.data:
            if isinstance(turn["utterance"], dict):
                turn["utterance"] = json.dumps(turn["utterance"])
    
    def reverse_parse_dialogue(self) -> None:
        """Convert utterances which are strings to dicts in-place."""
        for turn in self.data:
            if turn["role"] == Role.DOCTOR.value and isinstance(turn["utterance"], str):
                try:
                    turn["utterance"] = json.loads(turn["utterance"])
                except json.decoder.JSONDecodeError:
                    print(f"Failed to parse utterance: {turn['utterance']}")
                    raise json.decoder.JSONDecodeError

    def log(self) -> None:
        """Print the dialogue to stdout."""
        for turn in self.data:
            print(turn["role"] + ": " + turn["utterance"])

    def save_dialogue(self, save_path: Path, is_json: bool) -> None:
        if is_json:
            self.reverse_parse_dialogue()
        save_path.write_text(json.dumps(self.data, indent=4))
        # reset dialogue state
        if is_json:
            self.parse_dialogue()

class ZeroShotDRCoTDialogue(Dialogue):

    def __init__(self, data: list[dict[str, str]] = []):
        super().__init__(data)

    def text(self) -> str:
        sents = list()
        for turn in self.data:
            role_text = turn["role"][0].upper() + turn["role"][1:]
            if turn["role"] == Role.DOCTOR.value:
                utterance_obj = json.loads(turn["utterance"])
                if "question_to_ask" in utterance_obj:
                    utterance = utterance_obj["question_to_ask"]
                elif "diagnosis" in utterance_obj:
                    utterance = utterance_obj["diagnosis"]
                else:
                    raise ValueError("either 'question_to_ask' or 'diagnosis' should be in the utterance_obj")
            elif turn["role"] == Role.PATIENT.value:
                utterance = turn["utterance"]
            else:
                raise ValueError("role should be either doctor or patient")
            sent = f"{role_text}: {utterance}"
            sents.append(sent)
        return "\n".join(sents)

    def save_dialogue(self, save_path: Path, is_json: bool) -> None:
        data = copy.deepcopy(self.data)
        for i, turn in enumerate(data):
            if (turn["role"] == Role.DOCTOR.value) and isinstance(turn["utterance"], str):
                data[i]["utterance"] = json.loads(data[i]["utterance"])
        save_path.write_text(json.dumps(data, indent=4))

class MultiStageDialogue(Dialogue):
    """The dialogue between the PatientBot and the DoctorBot in the multi-stage DR-CoT setting.

    Format:
    [
        {
            "role": PATIENT,
            "utterance": str  # response to the doctor's question
        }
        {
            "role": DOCTOR, # Role.value
            "utterance": {
                "extracted_finding": str,
                "findings_summary": str,
                "ranked_ddx": list[str],
                "question": str
            }
        },
        ...
    ]
    """

    def __init__(self, data: list[dict[str, str]] = []):
        self.data = data
        self.last_question = None

    def add_utterance(self, role: Role, utterance: str):
        # raise an error to indicate that this is deprecated in MultiStageDialogue
        raise NameError("Use add_patient_utterance() or add_doctor_utterance() instead.")

    def add_patient_utterance(
        self,
        utterance: str
    ) -> None:
        """Add a patient utterance to the dialogue."""
        self.data.append({
            "role": Role.PATIENT.value,
            "utterance": utterance
        })

    def add_doctor_utterance(
        self,
        s: str,
        ss: str,
        ddx: list[str],
        question: str
    ) -> None:
        """Add a doctor utterance to the dialogue."""
        self.data.append({
            "role": Role.DOCTOR.value,
            "utterance": {
                "extracted_finding": s,
                "findings_summary": ss,
                "ranked_ddx": ddx,
                "question": question
            }
        })
        self.last_question = question

    def text(self) -> str:
        sents = list()
        for turn in self.data:
            if turn["role"] == Role.PATIENT.value:
                utter = turn["utterance"]
            elif turn["role"] == Role.DOCTOR.value:
                utter = turn["utterance"]["question"]
            else:
                raise ValueError("Invalid role.")
            sents.append(f"{turn['role']}: {utter}")
        return '\n'.join(sents)

    def save_dialogue(self, save_path: Path, is_json: bool) -> None:
        save_path.write_text(json.dumps(self.data, indent=4))
