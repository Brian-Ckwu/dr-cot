import re
import json
from pathlib import Path

class Context(object):

    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    def text(self) -> str:
        """Return the text of the context."""
        return self.raw_text

class PatientContext(Context):
    delimiter = "_@_"
    release_evidences = json.loads((Path(__file__).parent.parent.parent / "ddxplus/release_evidences.json").read_bytes())
    release_conditions = json.loads((Path(__file__).parent.parent.parent / "ddxplus/release_conditions.json").read_bytes())
    evidence2desc = json.loads((Path(__file__).parent.parent.parent / "ddxplus/our_evidences_to_qa_v2.json").read_bytes())
    desc_field = "affirmative_en"
    
    def __init__(
        self,
        sex: str,
        age: int,
        initial_evidence: str,
        evidences: list[str]
    ):
        if sex not in ['M', 'F']:
            raise ValueError(f"Sex should be either 'M' or 'F'")
        self._sex = sex
        self._age = age
        self._initial_evidence = initial_evidence
        self._parsed = self._parse_evidences(evidences)
        
    def text(self) -> str:
        """Convert parsed evidences to text representation."""
        sents = list()
        # Sex & age
        sex_name = 'Male' if self._sex == 'M' else 'Female'
        sent = f"Sex: {sex_name}, Age: {str(self._age)}"
        sents.append(sent)
        
        # binary
        for key in self._parsed['B']:
            sent = self.evidence2desc[key][self.desc_field]
            sents.append(f"- {sent}")
        # categorical
        for key, value in self._parsed['C'].items():
            sent = self.evidence2desc[key][self.desc_field]
            value = str(value)
            if value in self.release_evidences[key]["value_meaning"]:
                value = self.release_evidences[key]["value_meaning"][str(value)]["en"]
            sent = re.sub(pattern=r"\[choice\]", string=sent, repl=value)
            sents.append(f"- {sent}")
        # multichoice
        for key, values in self._parsed['M'].items():
            header = self.evidence2desc[key][self.desc_field]
            sents.append(f"- {header}")
            for value in values:
                item = self.release_evidences[key]["value_meaning"][value]["en"]
                sents.append(f"* {item}")
        
        return '\n'.join(sents)
        
    def _parse_evidences(self, evidences: list[str]) -> dict:
        """
            Parse the evidences into the following format:
            {
                "B": [key1, key2, ...],
                "C": {
                    key1: value1,
                    key2: value2,
                    ...
                },
                "M": {
                    key1: [value1, value2, ...],
                    key2: [value1, value2, ...],
                    ...
                }
            }
        """
        d = {'B': [], 'C': {}, 'M': {}}
        for evidence in evidences:
            evidence = evidence.split(self.delimiter)
            evidence_name = evidence[0]
            if evidence_name not in self.release_evidences:
                raise KeyError(f"There is no evidence called {evidence_name}")
            data_type = self.release_evidences[evidence_name]["data_type"]
            
            if len(evidence) == 1: # binary
                if data_type != 'B':
                    raise ValueError(f"The date_type of evidence {evidence_name} should be binary (B). Please check!")
                d[data_type].append(evidence_name)
                
            elif len(evidence) == 2: # categorical or multichoice
                if data_type not in ['C', 'M']:
                    raise ValueError(f"The date_type of evidence {evidence_name} should be either categorical (C) or multichoice (M). Please check!")
                evidence_value = evidence[1]
                if evidence_value in ["N", "NA"]: # the bug in the DDxPlus dataset
                    continue
                if data_type == 'C': # categorical
                    d[data_type][evidence_name] = evidence_value
                else: # multichoice
                    d[data_type][evidence_name] = d[data_type].get(evidence_name, []) + [evidence_value]
                    
            else:
                raise ValueError(f"After spliiting with {self.delimiter}, the length of {evidence} should be either 1 or 2.")
        
        return d
    
    @property
    def initial_evidence(self) -> str:
        return self.evidence2desc[self._initial_evidence][self.desc_field]
    
    @property
    def basic_info(self) -> str:
        return f"I am a {self._age}-year-old {'male' if self._sex == 'M' else 'female'}."

class DoctorContext(Context):

    def __init__(self):
        pass

if __name__ == "__main__":
    from data import DDxDataset

    csv_path = "../../ddxplus/release_test_patients.csv"
    pathology_info_path = "../../ddxplus/release_conditions.json"
    evidences_info_path = "../../ddxplus/our_evidences_to_qa_v2.json"

    dataset = DDxDataset(csv_path, pathology_info_path, evidences_info_path)
    print("Dataset loaded.")
    indices = [13]
    for index in indices:
        pat = dataset.df.iloc[index]
        pat_context = PatientContext(
            sex=pat.SEX,
            age=pat.AGE,
            initial_evidence=pat.INITIAL_EVIDENCE,
            evidences=pat.EVIDENCES
        )
        print(pat_context.text())
        print()
        print(pat_context.initial_evidence)
        print()
        print(pat_context.basic_info)
        print()
