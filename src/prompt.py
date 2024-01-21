import json
from pathlib import Path
from abc import ABC, abstractmethod

from model import PaLM2Model

class Prompt(ABC):
    ITEM_SEPARATOR = "; "
    LINE_SEPARATOR = "\n"
    SHOT_SEPARATOR = "\n\n\n"

    def __init__(
        self,
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        self.instruction = instruction
        self.shots = shots
        self.minimal_template = self.get_minimal_template()
        self.semantic_template = self.get_semantic_template()

    @abstractmethod
    def get_minimal_template(self) -> str:
        """Returns the prompt text built by minimal template."""
        raise NotImplementedError

    @abstractmethod
    def get_semantic_template(self) -> str:
        """Returns the prompt text built by semantic template (containing semantic cues)."""
        raise NotImplementedError

class SymptomExtractorPrompt(Prompt):
    QUESTION_CUE = "Question: "
    ANSWER_CUE = "Answer: "
    SYMPTOM_CUE = "Symptom: "

    def __init__(
        self,
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        super().__init__(instruction, shots)

    def get_minimal_template(self) -> str:
        """Returns the prefix prompt text built by minimal template.

        Format:
            [Instruction (optional)]
            <Shot_separator>
            [Question_1]
            [Answer_1]
            [Symptom_1]
            <Shot_separator>
            ...
            <Shot_separator>
            [Question_k]
            [Answer_k]
            [Symptom_k]
            <Shot_separator>
            [Question_test]
            [Answer_test]
        """
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                chunk = self.LINE_SEPARATOR.join(
                    [shot["question"],
                     shot["answer"],
                     shot["symptom"]]
                )
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)

    def get_minimal_prompt(self, q: str, a: str) -> str:
        chunks = [
            self.minimal_template,
            self.LINE_SEPARATOR.join([q, a])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

    def get_semantic_template(self) -> str:
        """Returns the prefix prompt text built by semantic template (containing semantic cues)."""
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                chunk = self.LINE_SEPARATOR.join(
                    [self.QUESTION_CUE + shot["question"],
                     self.ANSWER_CUE + shot["answer"],
                     self.SYMPTOM_CUE + shot["symptom"]]
                )
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)

    def get_semantic_prompt(self, q: str, a: str) -> str:
        chunks = [
            self.semantic_template,
            self.LINE_SEPARATOR.join([
                self.QUESTION_CUE + q,
                self.ANSWER_CUE + a
            ])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

class DDXPredictorPrompt(Prompt):
    POSITIVE_CUE = "Positive clinical findings: "
    NEGATIVE_CUE = "Negative clinical findings: "
    DDX_CUE = "Ranked differential diagnosis: "

    def __init__(
        self,
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        super().__init__(instruction, shots)

    def get_minimal_template(self) -> str:
        """Returns the prefix prompt text built by minimal template.

        Format:
            [Instruction (optional)]
            <Shot_separator>
            [Symptoms_1]
            [DDX_1]
            <Shot_separator>
            ...
            <Shot_separator>
            [Symptoms_k]
            [DDX_k]
            <Shot_separator>
            [Symptoms_test]
        """
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                pos = self.ITEM_SEPARATOR.join(shot["positive"]) if shot["positive"] else "None"
                neg = self.ITEM_SEPARATOR.join(shot["negative"]) if shot["negative"] else "None"
                ddx = self.ITEM_SEPARATOR.join(shot["ddx"])
                chunk = self.LINE_SEPARATOR.join([pos, neg, ddx])
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)        

    def get_minimal_prompt(self, pos: list[str], neg: list[str]) -> str:
        chunks = [
            self.minimal_template,
            self.LINE_SEPARATOR.join([
                self.ITEM_SEPARATOR.join(pos),
                self.ITEM_SEPARATOR.join(neg)
            ])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

    def get_semantic_template(self) -> str:
        """Returns the prefix prompt text built by semantic template (containing semantic cues)."""
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                pos = self.ITEM_SEPARATOR.join(shot["positive"]) if shot["positive"] else "None"
                neg = self.ITEM_SEPARATOR.join(shot["negative"]) if shot["negative"] else "None"
                ddx = self.ITEM_SEPARATOR.join(shot["ddx"])
                chunk = self.LINE_SEPARATOR.join([
                    self.POSITIVE_CUE + pos,
                    self.NEGATIVE_CUE + neg,
                    self.DDX_CUE + ddx
                ])
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)

    def get_semantic_prompt(self, pos: list[str], neg: list[str]) -> str:
        chunks = [
            self.semantic_template,
            self.LINE_SEPARATOR.join([
                self.POSITIVE_CUE + self.ITEM_SEPARATOR.join(pos),
                self.NEGATIVE_CUE + self.ITEM_SEPARATOR.join(neg)
            ])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

class QuestionGeneratorPrompt(Prompt):
    POSITIVE_CUE = "Positive clinical findings: "
    NEGATIVE_CUE = "Negative clinical findings: "
    DDX_CUE = "Ranked differential diagnosis: "
    QUESTION_CUE = "Question to narrow down the differential diagnosis: "

    def __init__(
        self,
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        super().__init__(instruction, shots)

    def get_minimal_template(self) -> str:
        """Returns the prefix prompt text built by minimal template.

        Format:
            [Instruction (optional)]
            <Shot_separator>
            [Symptoms_1]
            [DDX_1]
            [Q_1]
            <Shot_separator>
            ...
            <Shot_separator>
            [Symptoms_k]
            [DDX_k]
            [Q_k]
            <Shot_separator>
            [Symptoms_test]
            [DDX_test]
        """
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                pos = self.ITEM_SEPARATOR.join(shot["positive"]) if shot["positive"] else "None"
                neg = self.ITEM_SEPARATOR.join(shot["negative"]) if shot["negative"] else "None"
                ddx = self.ITEM_SEPARATOR.join(shot["ddx"])
                q = shot["q"]
                chunk = self.LINE_SEPARATOR.join([pos, neg, ddx, q])
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)        

    def get_minimal_prompt(self, pos: list[str], neg: list[str], ddx: list[str]) -> str:
        chunks = [
            self.minimal_template,
            self.LINE_SEPARATOR.join([
                self.ITEM_SEPARATOR.join(pos),
                self.ITEM_SEPARATOR.join(neg),
                self.ITEM_SEPARATOR.join(ddx)
            ])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

    def get_semantic_template(self) -> str:
        """Returns the prefix prompt text built by semantic template (containing semantic cues)."""
        chunks = list()
        # Instruction
        if self.instruction:
            chunks.append(self.instruction)
        # Few-shot examples
        if self.shots:
            for shot in self.shots:
                pos = self.ITEM_SEPARATOR.join(shot["positive"]) if shot["positive"] else "None"
                neg = self.ITEM_SEPARATOR.join(shot["negative"]) if shot["negative"] else "None"
                ddx = self.ITEM_SEPARATOR.join(shot["ddx"])
                q = shot["q"]
                chunk = self.LINE_SEPARATOR.join([
                    self.POSITIVE_CUE + pos,
                    self.NEGATIVE_CUE + neg,
                    self.DDX_CUE + ddx,
                    self.QUESTION_CUE + q
                ])
                chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)

    def get_semantic_prompt(self, pos: list[str], neg: list[str], ddx: list[str]) -> str:
        chunks = [
            self.semantic_template,
            self.LINE_SEPARATOR.join([
                self.POSITIVE_CUE + self.ITEM_SEPARATOR.join(pos),
                self.NEGATIVE_CUE + self.ITEM_SEPARATOR.join(neg),
                self.DDX_CUE + self.ITEM_SEPARATOR.join(ddx)
            ])
        ]
        return self.SHOT_SEPARATOR.join(chunks) + self.LINE_SEPARATOR

# Unit tests
if __name__ == "__main__":
    llm = PaLM2Model(config={
        "model": "models/text-bison-001",
        "temperature": 0.0,
        "candidate_count": 1,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 100,
        "stop_sequences": [SymptomExtractorPrompt.LINE_SEPARATOR]
    })
    qg_prompt_text = json.loads(Path("../prompts/doctor/multistage/question_generator.json").read_text())
    qg_prompt = QuestionGeneratorPrompt(**qg_prompt_text)

    pos = ["cough"]
    neg = ["dyspnea", "fever", "sore throat", "heartburn or acid reflux"]
    ddx = [
        "Bronchospasm / acute asthma exacerbation",
        "Allergic sinusitis",
        "Acute COPD exacerbation / infection",
        "GERD"
    ]
    qg_minimal_prompt = qg_prompt.get_minimal_prompt(pos, neg, ddx)
    qg_semantic_prompt = qg_prompt.get_semantic_prompt(pos, neg, ddx)

    minimal_res = llm.generate(prompt=qg_minimal_prompt)
    semantic_res = llm.generate(prompt=qg_semantic_prompt)
    print(f"Minimal response: {minimal_res}")
    print(f"Semantic response: {semantic_res}")