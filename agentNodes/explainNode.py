import os
from models.explain import Explain
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

parser = PydanticOutputParser(pydantic_object = Explain)

async def explain(state: AgentState) -> AgentState:
    """
    Explains the medical triage in a human friendly language that can be understood by the user
    """
    prompt = ChatPromptTemplate.from_template(
        """
        You are the Patient Explanation Agent.
        Your job is to translate the clinical triage decision into a simple, friendly message that a patient can easily understand. 
        You do not give medical treatment. You explain, reassure, and guide.

        You MUST use:
        - The triage result and clinical rationale  
        - The structured intake summary  
        - Any RAG evidence that supports the context  

        Your tone:
        - Calm, supportive, clear  
        - No jargon  
        - No diagnoses  
        - No medical instructions requiring a professional  
        - Focus on clarity and reassurance  

        What to produce:

        1. Message
        A short paragraph (4–6 sentences) explaining:
        - what the triage result means  
        - why their symptoms may need attention  
        - what the next reasonable step is  
        - reassurance without minimizing their concern  

        2. follow_up_steps  
        A short list of practical next steps the patient can safely do, such as:  
        - monitor symptoms  
        - drink water if appropriate  
        - keep track of changes  
        - prepare information for a clinician  
        Never give medication names, dosages, or treatment instructions.

        3. when_to_escalate
        A list of warning signs that mean the patient should seek higher-level help.  
        These should come from the triage level and red-flag criteria.  
        Examples:  
        - symptoms suddenly worsen  
        - new severe pain develops  
        - difficulty breathing starts  
        - confusion or fainting  

        Input Provided:
        - Intake Summary: {intake}
        - Triage Result: {triage}
        - Retrieved Evidence: {evidence}

        Output Format:
        You must follow this exact JSON schema:
        {format_instructions}

        Keep the explanation understandable and human.

        """
    ).partial(format_instructions = parser.get_format_instructions())

    chain = prompt | llm | parser
    response = await chain.ainvoke({
        'intake': state.refined_input,
        'triage': state.triage,
        'evidence': state.retrieve
    })

    return {'explainable_pitch': response}