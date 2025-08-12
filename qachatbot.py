import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Page config
st.set_page_config(page_title="Simple Langchain Chat with Groq")

# Title
st.title("Simple Langchain Chat with Groq")
st.markdown("Learn LangChain basics with Groq's Ultra-fast inference!")

with st.sidebar:
    st.header("Settings")

    # API Key
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get Free API Key at console.groq.com"
    )

    # Model Selection
    model_name = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "gemma2-9b-it"],  # Removed whisper-large-v3 (audio model)
        index=0
    )

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize LLM
@st.cache_resource(show_spinner=False)
def get_chain(api_key, model_name):
    if not api_key:
        return None
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant powered by Groq. Answer questions clearly and concisely."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain

# get chain
chain = get_chain(api_key, model_name)

if not chain:
    st.warning("Please enter your Groq API Key in the sidebar to start chatting!")
    st.markdown("[Get your free API Key here](https://console.groq.com)")
else:
    # Display the chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # chat input
    question = st.chat_input("Ask me anything")
    if question:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # Stream response from Groq
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response)
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")