class Context(object):

    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def text(self) -> str:
        """Return the text of the context."""
        return self.raw_text

class PatientContext(Context):

    def __init__(self):
        pass

class DoctorContext(Context):

    def __init__(self):
        pass