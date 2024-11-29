import json
import os
from datetime import datetime
from itertools import count

from dotenv import load_dotenv
import streamlit as st
from langchain_core.language_models.fake_chat_models import FakeChatModel
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


if "c" not in st.session_state:
    initial_state()


# 1) Create the columns with the setting options

st.write("LLM Settings:")
columns = st.columns(4)

MODEL_OPTIONS = ("gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "develop-debugging")
CHAIN_TYPES = ("Top Probability", "High Temperature")

with columns[0]:
    model_name = st.selectbox("Model name", MODEL_OPTIONS)
with columns[1]:
    chain_type = st.radio("Chain", CHAIN_TYPES)

if chain_type == "Top Probability":
    with columns[2]:
        temperature = st.number_input(
            "temperature", min_value=0.0, max_value=0.8, value="min", step=0.1
        )
        top_p = st.number_input(
            "top_p", min_value=0.1, max_value=1.0, value=0.2, step=0.1
        )

elif chain_type == "High Temperature":
    with columns[2]:
        temperature = st.number_input(
            "temperature", min_value=0.8, max_value=2.0, value=1.0, step=0.1
        )
        top_p = st.number_input(
            "top_p", min_value=0.1, max_value=1.0, value="min", step=0.1
        )


with columns[3]:
    memory = st.checkbox("Write Memory")
    read_only = st.checkbox("Read Only Memory")
    system_message = st.checkbox("System Message")


# 2) Create the input boxes for the user and system messages

if system_message and model_name not in ["o1-preview", "o1-mini"]:
    system_input = st.text_area(
        "System message:", placeholder="Enter your prompt here...", key="system_input"
    )

user_input = st.text_area(
    f"Turn {st.session_state.current_turn + 1} input:",
    placeholder="Enter your prompt here...",
    key="user_input",
)

# 3) Create the model instance
# Prompt
prompt_messages = [
    MessagesPlaceholder(variable_name="messages"),
]

if system_message and model_name not in ["o1-preview", "o1-mini"]:
    prompt_messages.insert(0, SystemMessage(content=system_input))

prompt_template = ChatPromptTemplate.from_messages(prompt_messages)

# Model
if model_name != "develop-debugging":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please, configure the OpenAI API key (OPENAI_API_KEY).")
    else:
        # Handling edge cases
        if model_name in ["o1-preview", "o1-mini"]:
            temperature = 1.0
            top_p = None
        llm_chain = prompt_template | ChatOpenAI(
            model=model_name, api_key=api_key, temperature=temperature, top_p=top_p
        )

else:
    llm_chain = prompt_template | FakeChatModel()


# 4) Send the request to the model

def _submit_turn():
    "Updates the current Streamlit Session State"
    st.session_state.user_request = st.session_state.user_input
    # NOTE: actualizar ´c´ implica actualizar st.session_state.user_input
    st.session_state.current_turn = next(st.session_state.c)
    st.session_state.chat_history["turn_" + str(st.session_state.current_turn)] = {}


request_confirmation = st.button("Submit", on_click=_submit_turn)

if request_confirmation:
    st.markdown("Request sent to the model - Please wait...")
    messages = []

    # Load previous conversation turns
    if 1 < st.session_state.current_turn and memory:
        latest_turns = list(st.session_state.chat_history.keys())[:-1]
        for turn in latest_turns:
            if st.session_state.chat_history[turn]["metadata"]["memory"]:
                messages += [
                    HumanMessage(
                        content=st.session_state.chat_history[turn][
                            "human"
                        ]
                    ),
                    AIMessage(
                        content=st.session_state.chat_history[turn]["ai"]
                    ),
                ]

    messages += [HumanMessage(content=st.session_state.user_request)]

    # Execute the model
    response = llm_chain.invoke({"messages": messages})
    st.session_state.output = response.content

    # Save the chat history
    st.session_state.chat_history["turn_" + str(st.session_state.current_turn)] = {
        "human": st.session_state.user_request,
        "ai": st.session_state.output,
        "metadata": {
            "model_name": model_name,
            "chain": chain_type,
            "temperature": temperature,
            "top_p": top_p,
            "memory": memory if not read_only else False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system": ""
        },
    }
    if system_message and model_name not in ["o1-preview", "o1-mini"]:
        st.session_state.chat_history[
            "turn_" + str(st.session_state.current_turn)
        ].update(
            {
                "system": system_input,
            }
        )


# 5) Display the output

st.markdown(st.session_state.output)


# 6) Session settings

@st.dialog("Do you want to reset the session?")
def _reset():
    if st.button("Confirm"):
        initial_state()
        st.rerun()


if st.checkbox("Session Settings"):
    st.download_button(
        "Download Chat History",
        data=json.dumps(st.session_state.chat_history, ensure_ascii=False).encode(
            "utf-8"
        ),
        file_name="chat_history.json",
    )

    st.button("Reset Session", on_click=_reset)
