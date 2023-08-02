class Context(object):

    def __init__(self):
        pass

    def text(self) -> str:
        """Return the text of the context."""
        raise NotImplementedError

class PatientContext(Context):

    def __init__(self):
        pass

class DoctorContext(Context):

    def __init__(self):
        pass