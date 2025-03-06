#!/usr/bin/env python3
import json
import os
from datetime import datetime
from itertools import count

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

st.title("LLM playground")
st.header("Freestyle Prompting")

# SETTING INITIAL SESSION STATE
def initial_state():
    st.session_state.c = count()
    st.session_state.current_turn = next(st.session_state.c)
    st.session_state.chat_history = {}
    st.session_state.output = ""

# Initialize state if not defined
if "c" not in st.session_state:
    initial_state()

# 1) Create the columns for the setting options
st.write("LLM Settings:")
columns = st.columns(4)

MODEL_OPTIONS = ("gpt-4o", "gpt-4o-mini", "o1", "o3-mini", "develop-debugging")
CHAIN_TYPES = ("Top Probability", "High Temperature")

with columns[0]:
    model_name = st.selectbox("Model name", MODEL_OPTIONS)
with columns[1]:
    chain_type = st.radio("Chain", CHAIN_TYPES)

# Common input parameters
with columns[2]:
    if chain_type == "Top Probability":
        temperature = st.number_input("Temperature", min_value=0.0, max_value=0.8, value=0.0, step=0.1)
        top_p = st.number_input("Top_p", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    elif chain_type == "High Temperature":
        temperature = st.number_input("Temperature", min_value=0.8, max_value=2.0, value=1.0, step=0.1)
        top_p = st.number_input("Top_p", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

with columns[3]:
    memory = st.checkbox("Write Memory")
    read_only = st.checkbox("Read Only Memory")
    # Now allow System Message for all models including o1 and o3-mini
    system_message_enabled = st.checkbox("System Message")

# 2) Create the input boxes for user and system messages
# Removed exclusion condition so system message input is always available when enabled
if system_message_enabled:
    system_input = st.text_area("System message:", placeholder="Enter your prompt here...", key="system_input")

user_input = st.text_area(
    f"Turn {st.session_state.current_turn + 1} input:",
    placeholder="Enter your prompt here...",
    key="user_input",
)

# 3) Create the prompt and model instance
prompt_messages = [MessagesPlaceholder(variable_name="messages")]

if system_message_enabled:
    # Insert system message at beginning of conversation
    prompt_messages.insert(0, SystemMessage(content=system_input))

prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

# Create LLM chain if not in debugging mode
if model_name != "develop-debugging":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please, configure the OpenAI API key (OPENAI_API_KEY).")
    else:
        # Handling reasoning models separately:
        if model_name in ["o1", "o3-mini"]:
            reasoning_effort = st.selectbox("Reasoning Effort", ("low", "medium", "high"))
            llm_chain = prompt_template | ChatOpenAI(
                model=model_name,
                api_key=api_key,
                reasoning_effort=reasoning_effort
            )
        else:
            llm_chain = prompt_template | ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=temperature,
                top_p=top_p
            )
else:
    model_response = st.text_area("Model response:", placeholder="Enter the expected response here...", key="model_response")

# 4) Sending the request to the model
def _submit_turn():
    "Updates the current Streamlit Session State"
    st.session_state.user_request = st.session_state.user_input
    # Updating counter; note that st.session_state.user_input remains intact
    st.session_state.current_turn = next(st.session_state.c)
    st.session_state.chat_history[f"turn_{st.session_state.current_turn}"] = {}

request_confirmation = st.button("Submit", on_click=_submit_turn)

if request_confirmation:
    st.markdown("Request sent to the model - Please wait...")
    messages = []

    # Load previous conversation turns if memory is enabled
    if st.session_state.current_turn > 1 and memory:
        # Exclude the current turn from history
        latest_turns_keys = list(st.session_state.chat_history.keys())[:-1]
        for turn in latest_turns_keys:
            # Use .get() to safely access metadata
            if st.session_state.chat_history[turn].get("metadata", {}).get("memory"):
                messages += [
                    HumanMessage(content=st.session_state.chat_history[turn].get("human", "")),
                    AIMessage(content=st.session_state.chat_history[turn].get("ai", "")),
                ]
    messages.append(HumanMessage(content=st.session_state.user_request))

    # Execute the model (if not develop-debugging)
    if model_name != "develop-debugging":
        response = llm_chain.invoke({"messages": messages})
        st.session_state.output = response.content
    else:
        st.session_state.output = model_response

    # Save the chat history
    chat_key = f"turn_{st.session_state.current_turn}"
    st.session_state.chat_history[chat_key] = {
        "human": st.session_state.user_request,
        "ai": st.session_state.output,
        "metadata": {
            "model_name": model_name,
            "chain": chain_type,
            "temperature": temperature,
            "top_p": top_p,
            "memory": memory if not read_only else False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": system_input if system_message_enabled else ""
        },
    }

# 5) Display output
st.markdown(st.session_state.output)

# 6) Session settings with option to reset or download chat history
@st.dialog("Do you want to reset the session?")
def _reset():
    if st.button("Confirm"):
        initial_state()
        st.rerun()

if st.checkbox("Session Settings"):
    st.download_button(
        "Download Chat History",
        data=json.dumps(st.session_state.chat_history, ensure_ascii=False).encode("utf-8"),
        file_name="chat_history.json",
    )
    st.button("Reset Session", on_click=_reset)
