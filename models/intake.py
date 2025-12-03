from pydantic import BaseModel, Field
from typing import List, Literal

class Intake(BaseModel):
    primary_symptom: str = Field(description = "Short description of the symptom")
    age: int = Field(description = 'Age of the patient')
    gender: Literal['Male','Female'] = Field(description = 'Gender of the patient')
    duration_hours: int = Field(description = "Duration of the symtpom in hours")
    severity: int = Field(ge = 1, le = 10, description = 'Severity of the symptom on a scale of 1-10')
    onset: Literal["sudden", "gradual", "intermittent", "constant"] = Field(description = 'At what point did it start')
    associated_symptoms: List[str] = Field(default_factory=list, description = 'Other symptoms that may be associated to the primary symptom')
    needs_clarrification: bool = Field(description = 'whether it needs clarrification or not')