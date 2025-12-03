import os
from models.triage import Triage
from models.agentState import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

parser = PydanticOutputParser(pydantic_object = Triage)

async def conduct_triage(state: AgentState) -> AgentState:
    """
    Sorts patients based on their severity condition
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are the Clinical Triage Agent. 
        Your job is to categorize the patient’s condition into one of the following levels:
        - immediate
        - urgent
        - serious
        - moderate
        - non-urgent

        You MUST use:
        1. The structured intake summary  
        2. Clarified details  
        3. Evidence retrieved from clinical guidelines (RAG)

        Your decision must be grounded strictly in:
        - WHO triage guidelines
        - emergency care rules
        - red-flag criteria
        - symptom clustering
        - age-specific severity cutoffs
        - evidence provided in the context

        ### Your responsibilities:
        • Analyze the structured intake data and RAG evidence.  
        • Detect red flags (severe pain, altered mental state, respiratory distress, uncontrolled bleeding, etc).  
        • Apply medical logic to choose the correct triage level.  
        • Never give treatment or diagnosis.  
        • Provide a concise clinical rationale referencing the evidence.

        ### Input Provided:
        - Intake summary: {intake}
        - Retrieved evidence: {evidence}

        ### Output Format:
        Follow this exact JSON structure:
        {format_instructions}

        ### Rules:
        • If ANY life-threatening sign appears → "immediate"  
        • If symptoms could worsen quickly or have moderate red flags → "urgent"  
        • If symptoms show clear medical concern but stable vitals → "serious"  
        • If symptoms are mild–moderate without red flags → "moderate"  
        • If symptoms are minor and no red flags → "non-urgent"

        Think like a triage nurse, not a doctor.  
        Base your decision only on the data provided.

        """
    ).partial(format_instructions = parser.get_format_instructions())

    chain = prompt | llm | parser
    response = await chain.ainvoke({
        'intake': state.refined_input,
        'evidence': state.retrieve
    })

    return {'triage': response}
