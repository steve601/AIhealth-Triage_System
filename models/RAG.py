from pydantic import BaseModel, Field

class Rag(BaseModel):
    evidence: str = Field(description = 'Short paragraph description of the medical protocals relevant to the patient symptoms')