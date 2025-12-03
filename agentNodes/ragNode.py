from models.RAG import Rag
from models.agentState import AgentState
from utils.text_cleaner import to_text, extract_text
from knowledge_base.embedder import create_embeddimgs
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1",
    temperature=0.0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
)

parser = PydanticOutputParser(pydantic_object=Rag)



async def get_relevant_evidence(state: AgentState) -> AgentState:
    """
    RAG Node: Filters retrieved evidence and outputs structured guideline snippets.
    """
    refined_text = to_text(state.refined_input)

    retriever = create_embeddimgs()

    # Get relevant documents
    docs1 = retriever.invoke(refined_text)

    # Extract only raw text content
    context1 = extract_text(docs1)

    prompt = ChatPromptTemplate.from_template("""
        You are the RAG Retrieval Agent.

        Your responsibilities:
        - You DO NOT generate medical advice.
        - You DO NOT create new medical information.
        - You only filter and return relevant guideline snippets already retrieved from the vector database.

        You will receive:
        1. Patient intake data:
        {input_data}

        2. Retrieved guideline chunks:
        [Context A]
        {ctx1}

        Your task:
        - Identify which snippets are the most clinically relevant for triage.
        - Return ONLY exact text extracted from the retrieved chunks.
        - Do NOT alter or rewrite the guideline content.
        - Do NOT add your own sentences.

        Return output in this JSON format:
        {format_instructions}
        """).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    response = await chain.ainvoke({
        "input_data": {
            "refined_input": state.refined_input,
        },
        "ctx1": context1,
    })

    return {"retrieve": response}
