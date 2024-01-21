import json
from pathlib import Path
from abc import ABC, abstractmethod

from model import PaLM2Model

class Prompt(ABC):
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

    def __init__(
        self,
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        super().__init__(instruction, shots)

    def get_minimal_template(self) -> str:
        """Returns the prompt text built by minimal template.

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
                raise NotImplementedError
        # Test example
        raise NotImplementedError
        chunks.append(chunk)
        return self.SHOT_SEPARATOR.join(chunks)        

class QuestionGeneratorPrompt(Prompt):

    def __init__(
        self,
        test_input: dict[str, str],
        instruction: str = None,
        shots: list[dict[str, str]] = None
    ):
        super().__init__(test_input, instruction, shots)

# Unit tests
if __name__ == "__main__":
    se_prompt_text = json.loads(Path("../prompts/doctor/multistage/symptom_extractor.json").read_text())
    se_prompt = SymptomExtractorPrompt(**se_prompt_text)

    # Test cases
    testcases = [
        {
            'q': "Do you have past history of allergy or asthma?",
            'a': "Yes.",
        },
        {
            'q': "Do you have past history of COPD?",
            'a': "No."
        },
        {
            'q': "How may I help you today?",
            'a': "I feel dizzy recently."
        }
    ]
    q, a = testcases[2]['q'], testcases[2]['a']
    se_minimal_prompt = se_prompt.get_minimal_prompt(q, a)
    se_semantic_prompt = se_prompt.get_semantic_prompt(q, a)

    # LLM
    llm = PaLM2Model(config={
        "model": "models/text-bison-001",
        "temperature": 0.0,
        "candidate_count": 1,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 100,
        "stop_sequences": [SymptomExtractorPrompt.LINE_SEPARATOR]
    })
    minimal_res = llm.generate(prompt=se_minimal_prompt)
    semantic_res = llm.generate(prompt=se_semantic_prompt)
    print(f"Response (from minimal prompt): {minimal_res}")
    print(f"Response (from semantic prompt): {semantic_res}")
