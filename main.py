import streamlit as st
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyBwZZBnDZXJUrqF7f-m-m0zxT3fYFtcQB8'


def get_vector_store():
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.load_local("faiss_index", embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    llm = GooglePalm(temperature=0.1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(),
                                                               memory=memory)
    return conversation_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:

            st.write("Bot: ", message.content)


def main():
    st.set_page_config("MedChatBot")
    st.header("Medical-ChatBot")
    with st.sidebar:
        st.title("Thank you for using our bot💖")
        st.subheader("PLEASE Click on Start")

        if st.button("Start"):
            with st.spinner("Processing"):
                vector_store = get_vector_store()
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("Done")
        st.subheader("For any queries , contact me at Kishore22705@gmail.com")

    config = "Hello , Lets start solving your issue"
    user_question = st.text_input("How can i assist you today?")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
