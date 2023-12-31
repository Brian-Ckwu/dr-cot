import json
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
