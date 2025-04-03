import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from streamlit_oauth import OAuth2Component
from dotenv import load_dotenv, find_dotenv

# Load environment variables (if using .env file for credentials)
load_dotenv(find_dotenv())

# Fake user database (replace with Firebase/SQL in production)
USER_CREDENTIALS = {
    "admin": "password123",
    "doctor": "medbot2025"
}

# Google OAuth setup
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_AUTH = OAuth2Component(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    token_url="https://oauth2.googleapis.com/token",
    redirect_uri="http://localhost:8501",
    scope="openid email profile"
)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Function to handle login/signup
def login_page():
    """Displays the login page with username/password and Google OAuth."""
    st.title("Welcome to MediBot!")
    st.subheader("Log in or Sign Up to access the chatbot, or continue as a guest.")

    # User input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        login_button = st.button("Login")
    with col2:
        signup_button = st.button("Sign Up")
    with col3:
        guest_button = st.button("Continue as Guest")

    # Login Logic
    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["messages"] = []  # Persist chat data for logged-in users
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    # Signup Logic
    if signup_button:
        if username and password:
            if username in USER_CREDENTIALS:
                st.error("Username already exists! Please choose another.")
            else:
                USER_CREDENTIALS[username] = password  # Fake sign-up (use a database in production)
                st.success("Account created! Please log in.")
        else:
            st.warning("Enter a username and password to sign up.")

    # Google OAuth login
    if st.button("Login with Google"):
        authorization_url, _ = GOOGLE_AUTH.get_login_url()
        st.session_state["google_auth_url"] = authorization_url
        st.markdown(f"[Click here to authenticate with Google]({authorization_url})")

    # Process OAuth response
    if "google_auth_url" in st.session_state:
        token = GOOGLE_AUTH.get_access_token()
        if token:
            st.session_state["logged_in"] = True
            st.session_state["username"] = "GoogleUser"
            st.session_state["messages"] = []
            st.success("Logged in with Google!")
            st.experimental_rerun()

    # Guest Mode
    if guest_button:
        st.session_state["logged_in"] = False
        st.session_state["messages"] = []  # Clear messages on every visit
        st.warning("You're using the chatbot in guest mode. Data will be erased when you leave.")
        st.experimental_rerun()

# Main chatbot function
def main():
    """Runs the chatbot if user is authenticated or a guest."""
    
    # Redirect to login page if not authenticated
    if "logged_in" not in st.session_state:
        login_page()
        return

    st.title("Ask Medical Chatbot!")
    st.text("This custom GPT leverages AI along an Encyclopedia entrusted by experienced Doctors.")
    st.text("Copyright © 2025 Prasad Sutar. All rights reserved.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Chat input
    prompt = st.chat_input("How can I assist you with your concern?")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
            Don’t provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
