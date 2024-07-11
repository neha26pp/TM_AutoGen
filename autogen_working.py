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
    config_list = [
        {
            "model": "gpt-4-turbo",
            "api_key": st.secrets["api_keys"]['OPENAI_API_KEY']
        }
    ]
    assert len(config_list) > 0
    return config_list

config_list = initialize_config()

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

llm_config = {"config_list": config_list, "timeout": 60, "temperature": 0.8, "seed": 1234}

@agentops.record_function('create_agents')
def create_agents():
    boss = autogen.UserProxyAgent(
        name="Boss",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="Reply `TERMINATE` when you are done comparing both the syllabi.",
        description="The boss who ask questions and give tasks.",
    )

    boss_aid = RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        default_auto_reply="Reply `TERMINATE` if the task is done.",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "code",
            "docs_path": ["syllabus1.txt", "syllabus2.txt"],
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "collection_name": "groupchat",
            "overwrite": True,
            "get_or_create": True,
        },
        code_execution_config=False,
        description="Assistant who has extra content retrieval power for solving difficult problems.",
    )

    transfer_credit_specialist = AssistantAgent(
        name="Transfer_Credit_Specialist",
        is_termination_msg=termination_msg,
        system_message='''You are a transfer credit specialist at a college. Your job is to compare 2 syllabi to decide if the courses should transfer or not based on the 3 criteria: topics covered, credits and grading criteria. Give a match percentage for each criteria (example, 20% similarity for topics covered). In the end, give a final match percentage using the provided weights and a detailed summary explaining your decision. Use latex to format math equations.''',
        llm_config=llm_config,
        description="Decides if there's a match or not.",
    )

    list_agent = AssistantAgent(
        name="list_agent",
        system_message="""Put all the percentage scores in a list as follows: [credits percentage, topics covered percentage, grading criteria percentage, final match percentage]. Do not return anything else.""",
        llm_config={
            "cache_seed": 41,
            "config_list": config_list,
            "temperature": 0,
        },
    )

    return boss, boss_aid, transfer_credit_specialist, list_agent

boss, boss_aid, transfer_credit_specialist, list_agent = create_agents()

@agentops.record_function('reset_agents')
def _reset_agents():
    boss.reset()
    boss_aid.reset()
    transfer_credit_specialist.reset()
    list_agent.reset()

@agentops.record_function('rag_chat')
def rag_chat(topics_covered_slider, credits_slider, grading_criteria_slider):
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[boss_aid, transfer_credit_specialist, list_agent],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    PROBLEM = f'''Will the CISC 110 course at Bucks County transfer in as an IST 110 course at Penn State? 
Use the following weights to calculate the final match percentage: {topics_covered_slider} for topics covered, {credits_slider} for credits and {grading_criteria_slider} for grading criteria.
Return the score percentages in a list. '''

    chat_result = boss_aid.initiate_chat(
        manager,
        message=boss_aid.message_generator,
        problem=PROBLEM,
        n_results=3,
    )
    return chat_result

@agentops.record_function('call_rag_chat')
def call_rag_chat():
    _reset_agents()

    @agentops.record_function('retrieve_content')
    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results"] = 3,
    ) -> str:
        boss_aid.n_results = n_results
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            _context = {"problem": message, "n_results": n_results}
            ret_msg = boss_aid.message_generator(boss_aid, None, _context)
        return ret_msg if ret_msg else message

    boss_aid.human_input_mode = "NEVER"

    for caller in [transfer_credit_specialist]:
        d_retrieve_content = caller.register_for_llm(
            description="retrieve content for code generation and question answering.", api_style="function"
        )(retrieve_content)

    for executor in [boss, boss_aid]:
        executor.register_for_execution()(d_retrieve_content)

    groupchat = autogen.GroupChat(
        agents=[boss, transfer_credit_specialist],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    boss.initiate_chat(
        manager,
        message=PROBLEM,
    )

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