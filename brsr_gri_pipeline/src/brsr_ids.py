"""
Reference BRSR ID scheme, taken directly from the linkage document's own
"About this linkage document" section:

    A1 refers to the first item under Section A: General disclosures
    P1 refers to Principle 1
    E1 refers to the first item under Essential indicators
    L1 refers to the first item under Leadership indicators

Table 1 (Summary Table) uses these composite IDs directly (e.g. "P6-E1").
Table 2 (Comprehensive Table) does NOT repeat these IDs -- it just has a
running "Sl. No" counter per section/principle/indicator-type block. We
reconstruct the composite ID from structural position while parsing Table 2,
using the state machine below, so both tables can be cross-validated
against each other in step 05.
"""

from dataclasses import dataclass


@dataclass
class ParserState:
    section: str = None          # "A", "B", or "C"
    subsection: str = None       # e.g. "I", "II"... within Section A/B
    principle: int = None        # 1-9, only set within Section C
    indicator_type: str = None   # "E" (Essential) or "L" (Leadership)
    counter: int = 0             # running counter within current block

    def reset_counter(self):
        self.counter = 0

    def next_id(self) -> str:
        self.counter += 1
        if self.section == "C":
            assert self.principle is not None and self.indicator_type is not None, \
                "Section C requires principle + indicator_type to be set"
            return f"P{self.principle}-{self.indicator_type}{self.counter}"
        elif self.section in ("A", "B"):
            return f"{self.section}{self.counter}"
        else:
            raise ValueError(f"Unhandled section state: {self.section}")


# Principle text -> number, for detecting "PRINCIPLE 6 Businesses should..." headers
PRINCIPLE_KEYWORDS = {
    1: "conduct and govern themselves with integrity",
    2: "goods and services in a manner that is sustainable and safe",
    3: "well-being of all employees",
    4: "respect the interests of and be responsive to all its stakeholders",
    5: "respect and promote human rights",
    6: "protect and restore the environment",
    7: "influencing public and regulatory policy",
    8: "inclusive growth and equitable development",
    9: "engage with and provide value to their consumers",
}
