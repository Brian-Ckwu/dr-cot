from bot import Role

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
    
    def __len__(self) -> int:
        """Return the number of utterances in the dialogue."""
        return len(self.data)
    
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