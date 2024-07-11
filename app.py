__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage
import tempfile
from datetime import timedelta
import os 
import autogen_working
import re
import pandas as pd
import ast
import autogen_chat_with_syllabus
import json 
import agentops

# Initialize AgentOps
AGENTOPS_API_KEY = st.secrets.get("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in Streamlit secrets")

agentops.init(AGENTOPS_API_KEY)

@agentops.record_function('initialize_firebase')
def initialize_firebase():
    firebase_config = st.secrets["firebase"]
    try:
        cred = credentials.Certificate({
            "type": firebase_config["type"],
            "project_id": firebase_config["project_id"],
            "private_key_id": firebase_config["private_key_id"],
            "private_key": firebase_config["private_key"].replace('\\n', '\n'),
            "client_email": firebase_config["client_email"],
            "client_id": firebase_config["client_id"],
            "auth_uri": firebase_config["auth_uri"],
            "token_uri": firebase_config["token_uri"],
            "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
            "client_x509_cert_url": firebase_config["client_x509_cert_url"]
        })
        firebase_admin.initialize_app(cred, {'storageBucket': 'carl-1667a.appspot.com'})
    except ValueError as e:
        pass

initialize_firebase()

st.title('TransferMaster')

col1, col2 = st.columns(2)

file1_url = None
file2_url = None

@agentops.record_function('upload_file_to_firebase')
def upload_file_to_firebase(file, local_base_name):
    if file:
        _, ext = os.path.splitext(file.name)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(file.read())
        temp_file.close()

        bucket = storage.bucket()
        blob = bucket.blob(file.name)
        blob.upload_from_filename(temp_file.name)

        url = blob.generate_signed_url(timedelta(days=365))
       
        local_name = local_base_name + ext
        local_path = os.path.join(".", local_name)
        blob.download_to_filename(local_path)

        return url, local_path
    return None, None

with col1:
    file1 = st.file_uploader("Upload first file", type=["txt"])
    if file1:
        file1_url = upload_file_to_firebase(file1, "syllabus1")
        st.write("First file uploaded.")

with col2:
    file2 = st.file_uploader("Upload second file", type=["txt"])
    if file2:
        file2_url = upload_file_to_firebase(file2, "syllabus2")
        st.write("Second file uploaded.")

topics_covered_slider = st.slider('Learning Objectives', 0, 100, 60)
credits_slider = st.slider('Textbook', 0, 100, 40)
grading_criteria_slider = st.slider('Grading Criteria', 0, 100, 10)

col3, col4 = st.columns([1, 1])

with col3:
    @agentops.record_function('compare_syllabi')
    def compare_syllabi():
        if file1_url and file2_url:
            chat_result = autogen_working.rag_chat(topics_covered_slider, credits_slider, grading_criteria_slider)
            
            output_text = chat_result.chat_history[1]["content"]
            st.write(output_text)

            scores_str = chat_result.chat_history[2]["content"]
            scores = ast.literal_eval(scores_str)
            print(scores)
            
            final_score = scores[3]
            st.markdown(f"**Final Score: {final_score}%**")
            first_three_scores = scores[:3]

            df = pd.DataFrame({
                'Scores': first_three_scores
            }, index=['Credits', 'Topics Covered', 'Grading Criteria'])

            st.bar_chart(df)
        else:
            st.error("Please upload both files to compare.")

    if st.button('Compare'):
        compare_syllabi()

    if st.button('Chat with Syllabus'):
        st.session_state.chat_open = True

@agentops.record_function('chat_with_syllabus')
def chat_with_syllabus():
    if 'chat_open' in st.session_state and st.session_state.chat_open:
        with st.expander("Chat with Syllabus"):
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'chat_input' not in st.session_state:
                st.session_state.chat_input = ""

            chat_input = st.text_area("Enter your message:", value=st.session_state.chat_input, key="chat_input_area")
            if st.button("Send", key="send_button"):
                chat_result = autogen_chat_with_syllabus.rag_chat(chat_input)
                response = chat_result.chat_history[1]["content"]
                
                st.session_state.chat_history.append(f"User: {chat_input}")
                st.session_state.chat_history.append(f"TransferMaster: {response}")

                st.session_state.chat_input = ""

            if st.session_state.chat_history:
                st.write("Chat History:")
                for message in st.session_state.chat_history:
                    st.write(message)

chat_with_syllabus()

@agentops.record_function('main')
def main():
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')