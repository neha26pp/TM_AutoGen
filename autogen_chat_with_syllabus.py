import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent
import chromadb
from typing_extensions import Annotated
import os
import dotenv
import streamlit as st
import agentops

# Initialize AgentOps
AGENTOPS_API_KEY = st.secrets.get("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in Streamlit secrets")

agentops.init(AGENTOPS_API_KEY)

@agentops.record_function('initialize_config')
def initialize_config():
    return [
        {
            "model": "gpt-4-turbo",
            "api_key": st.secrets["api_keys"]['OPENAI_API_KEY']
        }
    ]

config_list = initialize_config()

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

@agentops.record_function('create_assistant')
def create_assistant():
    return RetrieveAssistantAgent(
        name="assistant",
        system_message="You are given 2 course syllabi. Your job is to answer user's questions about the syllabi.",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
        },
    )

@agentops.record_function('create_ragproxyagent')
def create_ragproxyagent():
    return RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "code",
            "docs_path": ["syllabus1.txt",
                          "syllabus2.txt"],
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "collection_name": "groupchat",
            "overwrite": True,
            "get_or_create": True,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
        description="Assistant who has extra content retrieval power for solving difficult problems.",
    )

assistant = create_assistant()
ragproxyagent = create_ragproxyagent()

@agentops.record_function('rag_chat')
def rag_chat(user_question):
    chat_result = ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=user_question,
    )  
    return chat_result

@agentops.record_function('main')
def main():
    # This function can be used to run any initialization or main logic
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')