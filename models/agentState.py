from pydantic import BaseModel, Field
from typing import Optional
from models.intake import Intake
from models.RAG import Rag
from models.triage import Triage
from models.explain import Explain

class AgentState(BaseModel):
    raw_input: str = Field(description = 'In depth description from the user')
    refined_input: Optional[Intake] = Field(
        default = None,
        description = 'Refined user input into medical structured data'
    )
    retrieve: Optional[Rag] = Field(
        default = None,
        description = 'Medical guidlines and protocal related to the user symptoms'
    )
    triage: Optional[Triage] = Field(
        default = None,
        description = 'Final clinical decision'
    )
    explainable_pitch: Optional[Explain] = Field(
        default = None,
        description = 'Simple friendly language for the patient'
    )