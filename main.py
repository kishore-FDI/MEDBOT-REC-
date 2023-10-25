import streamlit as st
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from langchain import PromptTemplate

os.environ['GOOGLE_API_KEY'] = 'AIzaSyBwZZBnDZXJUrqF7f-m-m0zxT3fYFtcQB8'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try to answer in points.

Context: {context}
Question: {question}

Return the answer below with the explanation in simple words with an example and deep knowledge.
Answer with explanation in simple words:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# def get_vector_store():
#     embeddings = GooglePalmEmbeddings()
#     vector_store = FAISS.load_local("faiss_index", embeddings)
#     return vector_store


def retrieval_qa_chain(prompt,db):
    llm = GooglePalm(temperature=0.001)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=db.as_retriever(search_kwargs={"k": 10}),
                                                               memory=memory,
                                                               prompt=prompt)
    return conversation_chain


def qa_bot():
    embeddings = GooglePalmEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(qa_prompt, db)
    return qa


if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
# Define sidebar content
st.sidebar.title("Sample Questions")
st.sidebar.write("1. What are the symptoms of COVID-19?")
st.sidebar.write("2. How can I lower my blood pressure?")
st.sidebar.write("3. Tell me about diabetes management.")
st.sidebar.write("4. How to treat a common cold?")
st.sidebar.write("5. Describe the signs of a heart attack.")
st.sidebar.write("6. Contact Details:")
st.sidebar.write("   Email: kishore22705@gmail.com.com")


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:

            st.write("Bot: ", message.content)


def main():
    st.set_page_config("LLM-project")
    st.header("Medical-ChatBot")
    for message in st.session_state['conversation_history']:
        if message['role'] == 'user':
            st.text(f"User: {message['content']}")
        else:
            st.text(f"Bot: {message['content']}")

        # User input at the bottom
    user_query = st.text_input("Enter your medical query:")

    if st.button("Submit"):
        # Get the chatbot's response
        response = qa_bot()({'question': user_query})

        # Store the conversation history
        st.session_state['conversation_history'].append({"role": "user", "content": user_query})
        st.session_state['conversation_history'].append({"role": "assistant", "content": response['answer']})
        # Use st.empty() to update the response
        response_placeholder = st.empty()
        response_placeholder.text(f"Bot: {response['answer']}")


if __name__ == "__main__":
    main()
