from pydantic import BaseModel, Field
from typing import List

class Explain(BaseModel):
    message: str = Field(description = "Short paragraph user firendly message to the patient")
    follow_up_steps: List[str] = Field(description = "Follow up steps for the patient")
    when_to_escalate: List[str] = Field(description = "Symptoms when it's beyond your control")
    