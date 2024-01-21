import json
import unittest
from pathlib import Path

from model import PaLM2Model
from prompt import SymptomExtractorPrompt, DDXPredictorPrompt

class TestSymptomExtractorPrompt(unittest.TestCase):

    def setUp(self):
        self.llm = PaLM2Model(config={
            "model": "models/text-bison-001",
            "temperature": 0.0,
            "candidate_count": 1,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 100,
            "stop_sequences": [SymptomExtractorPrompt.LINE_SEPARATOR]
        })
        se_prompt_text = json.loads(Path("../prompts/doctor/multistage/symptom_extractor.json").read_text())
        self.se_prompt = SymptomExtractorPrompt(**se_prompt_text)

    def test_positive(self):
        q = "Do you have past history of allergy or asthma?"
        a = "Yes."
        se_minimal_prompt = self.se_prompt.get_minimal_prompt(q, a)
        se_semantic_prompt = self.se_prompt.get_semantic_prompt(q, a)
        minimal_res = self.llm.generate(prompt=se_minimal_prompt)
        minimal_pol, minimal_sym = self.se_prompt.parse_minimal_response(minimal_res)
        self.assertEqual(minimal_pol, "positive")
        self.assertEqual(minimal_sym, "allergy or asthma")
        semantic_res = self.llm.generate(prompt=se_semantic_prompt)
        semantic_pol, semantic_sym = self.se_prompt.parse_semantic_response(semantic_res)
        self.assertEqual(semantic_pol, "positive")
        self.assertEqual(semantic_sym, "allergy or asthma")

    def test_negative(self):
        q = "Do you have past history of COPD?"
        a = "No."
        se_minimal_prompt = self.se_prompt.get_minimal_prompt(q, a)
        se_semantic_prompt = self.se_prompt.get_semantic_prompt(q, a)
        minimal_res = self.llm.generate(prompt=se_minimal_prompt)
        minimal_pol, minimal_sym = self.se_prompt.parse_minimal_response(minimal_res)
        self.assertEqual(minimal_pol, "negative")
        self.assertEqual(minimal_sym, "COPD history")
        semantic_res = self.llm.generate(prompt=se_semantic_prompt)
        semantic_pol, semantic_sym = self.se_prompt.parse_semantic_response(semantic_res)
        self.assertEqual(semantic_pol, "negative")
        self.assertEqual(semantic_sym, "COPD history")

    def test_chief_complaint(self):
        q = "How may I help you today?"
        a = "I feel dizzy recently."
        se_minimal_prompt = self.se_prompt.get_minimal_prompt(q, a)
        se_semantic_prompt = self.se_prompt.get_semantic_prompt(q, a)
        minimal_res = self.llm.generate(prompt=se_minimal_prompt)
        minimal_pol, minimal_sym = self.se_prompt.parse_minimal_response(minimal_res)
        self.assertEqual(minimal_pol, "positive")
        self.assertEqual(minimal_sym, "dizziness")
        semantic_res = self.llm.generate(prompt=se_semantic_prompt)
        semantic_pol, semantic_sym = self.se_prompt.parse_semantic_response(semantic_res)
        self.assertEqual(semantic_pol, "positive")
        self.assertEqual(semantic_sym, "dizziness")

class TestDDXPredictorPrompt(unittest.TestCase):

    def setUp(self):
        self.llm = PaLM2Model(config={
            "model": "models/text-bison-001",
            "temperature": 0.0,
            "candidate_count": 1,
            "top_k": 40,
            "top_p": 0.95,
            "max_output_tokens": 100,
            "stop_sequences": [SymptomExtractorPrompt.LINE_SEPARATOR]
        })
        ddx_prompt_text = json.loads(Path("../prompts/doctor/multistage/ddx_predictor.json").read_text())
        self.ddx_prompt = DDXPredictorPrompt(**ddx_prompt_text)

    def test_ddx_set(self):
        ddx_set = {
            "Pneumonia",
            "Influenza",
            "GERD",
            "Bronchospasm / acute asthma exacerbation",
            "Acute COPD exacerbation / infection",
            "Allergic sinusitis"
        }
        pos = ["cough", "chest pain"]
        neg = ["fever"]
        ddx_semantic_prompt = self.ddx_prompt.get_semantic_prompt(pos, neg)
        semantic_res = self.llm.generate(prompt=ddx_semantic_prompt)
        ddx = self.ddx_prompt.parse_semantic_response(semantic_res)
        # check that each ddx is in the ddx set
        for d in ddx:
            self.assertIn(d, ddx_set)

if __name__ == "__main__":
    unittest.main()
