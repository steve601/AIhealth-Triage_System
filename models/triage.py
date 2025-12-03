from pydantic import BaseModel, Field
from typing import Literal

class Triage(BaseModel):
    triage: Literal["urgent", "immediate", "serious", "moderate","non-urgent"] = Field(description = 'Category of the triage')
    confidence: float = Field(ge = 0.01, le = 1.0, description = 'Confidence level of the triage')
    clinical_rationale: str = Field(description = 'Justification behind the clinical decision')
    