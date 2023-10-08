import streamlit as st
from firebase_admin import firestore
import os
from datetime import datetime
from utils import ai_bot, small_questionnaire, full_questionnaire, insurance_advisor, get_tokens, download_transcript
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch, FAISS
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


# @st.cache_resource(ttl="1h")
def get_document(uploaded_files, openai_api_key):
    """
    :param uploaded_files:
    :return:
    """
    text = ""
    for uploaded_file in uploaded_files:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n",
                                          chunk_size=1000,
                                          chunk_overlap=20,
                                          length_function=len)

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store


def new_qa():
    """
    entry point of the QA doc
    :return:
    """
    st.title("Reflexive AI")
    st.header("QA documents")

    if 'qa_doc_api_key_set' not in st.session_state:
            st.session_state.qa_doc_api_key_set = False
            st.session_state.qa_doc_uploaded = False

    if st.sidebar.button("Logout"):
        db = firestore.client()  # log in table
        obj = {"name": st.session_state["name"],
               "username": st.session_state["username"],
               "login_connection_time": st.session_state["login_connection_time"],
               "messages": st.session_state["messages"],
               "created_at": datetime.now()}
        doc_ref = db.collection(u'users_app').document()  # create a new document.ID
        doc_ref.set(obj)  # add obj to collection
        db.close()

        st.empty()  # clear page
        print(f"in logout from QA doc:{st.session_state}")
        for key in st.session_state.keys():
            print(f"{key} --> {st.session_state[key]}")

        # delete all keys
        for key in st.session_state.keys():
            del st.session_state[key]

        print(st.session_state)
        return None

    if st.sidebar.button("Download transcripts"):
        download_transcript()

    model = st.sidebar.selectbox(
        label=":blue[MODEL]",
        options=["gpt-3.5-turbo",
                 "ft:gpt-3.5-turbo-0613:osc:finetuned-v6:80vd3iOe",
                 "gpt-4"])

    systemprompt = st.sidebar.selectbox(
        label=":blue[AI Persona]",
        options=["Simple AI Assistant",
                 "mini questionnaire",
                 "full questionnaire",
                 "Insurance Advisor"])

    show_tokens = st.sidebar.radio(label=":blue[Display tokens]", options=('Yes', 'No'))

    # Set API key if not yet
    openai_api_key = st.sidebar.text_input(
        ":blue[API-KEY]",
        placeholder="Paste your OpenAI API key here",
        type="password")

    if openai_api_key:

        # openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = model

        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("Please upload PDF documents to continue.")
            st.stop()
        st.session_state.qa_doc_uploaded = True

        tmp_retriever = get_document(uploaded_files, openai_api_key)
        retriever = tmp_retriever.as_retriever()

        # Setup memory for contextual conversation
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        # Setup LLM and QA chain
        llm = ChatOpenAI(
            model_name=model, temperature=0, streaming=True, openai_api_key=openai_api_key
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True
        )

        if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message("How can I help you?")

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

            print(f"current message:{st.chat_message}")
            print(f"current messages:{msgs}")


# Run the Streamlit app
if __name__ == "__main__":
    print(f"in qa_docs.py, starting to run new_qa()")
    print(f"st.session_state: {st.session_state}")
    new_qa()
