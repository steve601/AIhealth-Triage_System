import os
from models.intake import Intake
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

parser = PydanticOutputParser(pydantic_object = Intake)

async def intake(state: AgentState) -> AgentState:
    """
    Understands the user's raw text input and converts it to a well medical structured data
    """
    prompt = ChatPromptTemplate.from_template(
        """ 
        You are an Intake Extraction Agent designed to read a patient’s complaint and convert it into structured, medically normalized data.
        You are not diagnosing. You are only extracting and organizing information.
        Your task:
            Given raw text {raw} from the user input, extract the following;
                -primary_symptom -> normalize to standard medical terminology
                -age -> age of the patient
                -gender -> gender of the patient
                -duration_hours -> convert any duration into hours
                -severity -> 1–10 scale (estimate if user uses descriptions like “mild,” “severe,” “unbearable”)
                -onset -> “sudden,” “gradual,” “unknown”
                -associated_symptoms -> list of additional symptoms mentioned
                -needs_clarification -> true if important information is missing, unclear or inconsistent
        Always extract based strictly on what the user said.
        If something is missing, return null or an empty list.
        Do NOT invent symptoms or make assumptions
        Format your entire response strictly as valid JSON matching this schema:
        {format_instructions}
        """
    ).partial(format_instructions = parser.get_format_instructions())

    chain = prompt | llm | parser
    response = await chain.ainvoke({'raw':state.raw_input})

    return {'refined_input':response}
