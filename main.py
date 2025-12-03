import streamlit as st
import asyncio
from models.agentState import AgentState
from graph.graph_builder import buid_graph

my_graph = buid_graph()

st.title('AI Patient Triage System')
st.markdown(
    "A web-based assistant that gathers a patient’s presenting complaint,usesclinical guidelines and patient context to assess urgency, and returns a triage "
    "recommendation."
    "Describe your medical condition, just feel free and don't forget to mention your age and gender too...)"
)

user_input = st.text_area('How do you feel?',height = 120)

if st.button('Conduct a triage...'):
    if not user_input.strip():
        st.error('Please describe your condition first!')
    else:
        placeholder = st.empty()
        state_container = {"final": None}

        async def run_stream():
            state = AgentState(raw_input = user_input)
            async for event in my_graph.astream_events(state):
                event_type = event["event"]
                data = event["data"]
                
                if event_type == "on_node_start":
                    placeholder.info(f"Running: {data['name']}")

                if event_type == "on_node_end":
                    placeholder.success(f"Finished: {data['name']}")

                if event_type == "on_chain_end":
                    state_container["final"] = data["output"]

        with st.spinner('Thinking...Conducting triage...'):
            asyncio.run(run_stream())

        result = state_container["final"]
        st.success('Triage complete!!')

        st.subheader('Your structured medical data')
        st.json(result["refined_input"].dict())

        st.subheader('Suggested triage')
        st.json(result["triage"].dict())

        st.subheader('AI pitch')
        st.json(result["explainable_pitch"].dict())
