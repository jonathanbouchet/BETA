import time
import logging
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from utils import *

# Create and configure logger
logging.basicConfig(
    filename="app.log",
    format='%(asctime)s %(message)s',
    filemode='w', )

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


def new_chatbot():
    """
    entry point of conversation
    :return:
    """
    # check a logout has not been issued already
    # TO DO: add in st.session_state a key for open_ai_key set true and check before setting the api key
    if not st.session_state["logout"] or st.session_state["authentication_status"] is True:

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if 'responses' not in st.session_state:
            st.session_state.responses = []
        if 'requests' not in st.session_state:
            st.session_state.requests = []
        if 'api_key_set' not in st.session_state:
            st.session_state.api_key_set = False

        # check a logout has not been issued already
        # if not st.session_state["logout"] or st.session_state["authentication_status"] is True:

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
            print(f"in logout from chatbot:{st.session_state}")
            st.session_state["logout"] = True
            st.session_state["name"] = None
            st.session_state["username"] = None
            st.session_state["authentication_status"] = None
            st.session_state["login_connection_time"] = None
            st.session_state["comment"] = []
            # st.session_state["authentication_status"] = False
            st.session_state['messages'] = []
            print(st.session_state)
            return st.session_state["logout"]

        st.title("Reflexive AI")
        st.header("Virtual Insurance Agent Accelerator")

        # Set API key if not yet
        # if st.session_state.api_key_set == False:
        openai_api_key = st.sidebar.text_input(
            ":blue[API-KEY]",
            placeholder="Paste your OpenAI API key here",
            type="password")

        MODEL = st.sidebar.selectbox(
            label=":blue[MODEL]",
            options=["gpt-3.5-turbo-16k-0613",
                     "gpt-3.5-turbo",
                     "gpt-3.5-turbo-16k",
                     "gpt-3.5-turbo-0613",
                     "gpt-4"])

        show_tokens = st.sidebar.radio(label=":blue[Display tokens]", options=('Yes', 'No'))

        if openai_api_key:
            st.session_state.api_key_set = True

            def get_conversation_string() -> str:
                """

                :return:
                """
                conversation_string = ""
                for i in range(len(st.session_state['responses']) - 1):
                    conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
                    conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
                return conversation_string

            llm = ChatOpenAI(model_name=MODEL, openai_api_key=openai_api_key, temperature=0)
            if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory = ConversationBufferWindowMemory(
                    k=10,
                    return_messages=True
                )

            system_msg_template = SystemMessagePromptTemplate.from_template(
                template=template_insurance()
            )

            human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
            prompt_template = ChatPromptTemplate.from_messages(
                [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
            )
            conversation = ConversationChain(
                memory=st.session_state.buffer_memory,
                prompt=prompt_template,
                llm=llm, verbose=True
            )

            # greeting message from the assistant
            with st.chat_message("assistant"):
                greeting_msg = "how can I help you today ?"
                st.markdown(greeting_msg)
                # Add assistant response to chat history

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Accept user input
            if prompt := st.chat_input("type your message here"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                    if show_tokens == "Yes":
                        prompt_tokens = get_tokens(prompt, MODEL)
                        tokens_count = st.empty()
                        tokens_count.caption(f"""query used {prompt_tokens} tokens """)
                        logging.info(f"[user]:{prompt}, # tokens:{prompt_tokens}")

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    conversation_string = get_conversation_string()
                    response = conversation.predict(input=f"Query:\n{prompt}")
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                    message_placeholder.markdown(full_response)
                    if show_tokens == "Yes":
                        assistant_tokens = get_tokens(full_response, MODEL)
                        tokens_count = st.empty()
                        tokens_count.caption(f"""assistant used {assistant_tokens} tokens """)
                        logging.info(f"[assistant]:{full_response}, # tokens:{assistant_tokens}")

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                # check for user response right at the beginning by looking at any message non-empty
                # if st.session_state:
                #     if st.session_state['messages'][-1]['role'] == "assistant":
                #         download_transcript(st.session_state['messages'][-1]['content'])

                print(f"end of session\t{st.session_state.messages}")


# Run the main page
if __name__ == "__main__":
    print(f"in real_chatbot.py, starting to run main")
    new_chatbot()
