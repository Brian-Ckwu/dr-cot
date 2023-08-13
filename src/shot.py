from dataclasses import dataclass
from .context import Context
from .dialogue import Dialogue

@dataclass
class Shot(object):
    context: Context
    dialogue: Dialogue
