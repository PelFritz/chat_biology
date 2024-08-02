from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import streamlit as st


if not "chat_history" in st.session_state:
    st.session_state.chat_history = []
st.set_page_config(page_title="Chat Biology", page_icon=":robot_face")
st.title("Chat Biology")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


model = ChatOllama(model="llama3", temperature=0)


def get_response(query, chat_history):
    template = """
    You are a Biology assistant: Assist in answering the questions.
    
    chat history: {chat_history}
    human question: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    return chain.stream({"query": query, "chat_history": chat_history})


query = st.chat_input('Question:')
if query is not None and query != '':
    st.session_state.chat_history.append(HumanMessage(query))
    with st.chat_message('Human'):
        st.markdown(query)

    with st.chat_message('AI'):
        ai_response = st.write_stream(get_response(query, st.session_state.chat_history))
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))

