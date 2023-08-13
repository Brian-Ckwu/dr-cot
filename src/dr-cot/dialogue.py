import json
from typing import Union

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
        self.parse_dialogue(data)
        self.data = data
    
    def __len__(self) -> int:
        """Return the number of utterances in the dialogue."""
        return len(self.data)
    
    def add_utterance(
        self,
        role, # Role
        utterance: str
    ):
        """Add an utterance to the dialogue."""
        self.data.append({
            "role": role.value,
            "utterance": utterance
        })
    
    def parse_dialogue(self, data: list[dict[str, str]]) -> None:
        """Convert utterances which are dicts to strings in-space."""
        for turn in data:
            if isinstance(turn["utterance"], dict):
                turn["utterance"] = json.dumps(turn["utterance"])
