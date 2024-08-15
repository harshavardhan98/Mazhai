import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag_test import ChatPDF

st.set_page_config(page_title="MazhAi")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        print(user_text)
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
            ext = file.name.split('.')[-1]

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path, ext)
        os.remove(file_path)


def read_and_save_folder():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    folder_path = st.session_state["folder_uploader"]
    with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting files from {folder_path}"):
        st.session_state["assistant"].ingest_folder(folder_path)

def set_style():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    set_style()
    st.header("MazhAi - Chat with your codebase")

    st.subheader("Upload a folder containing Java files")
    folder_uploader = st.text_input(
        "Enter the folder path",
        key="folder_uploader",
        on_change=read_and_save_folder,
        label_visibility="collapsed",
    )

    st.text("")
    st.text("")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf", "java"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_area("Message", height=275, key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()