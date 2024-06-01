import autogen

from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent
import chromadb
from typing_extensions import Annotated
import os
import dotenv


config_list = [
  {
    "model": "gpt-4-turbo",
    "api_key": st.secrets["api_keys"]['OPENAI_API_KEY']
  }
]
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are given 2 course syllabi. Your job is to answer user's questions about the syllabi.",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": config_list,
    },
)

ragproxyagent = RetrieveUserProxyAgent(
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


def rag_chat(user_question):
    chat_result = ragproxyagent.initiate_chat(
        assistant, message=ragproxyagent.message_generator, problem=user_question,
    )  

    return chat_result