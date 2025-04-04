import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# Initialize tools
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun()

# App UI
st.title("🔍 Search Assistant")
st.write("Ask me anything and I'll search the web, arXiv, and Wikipedia for answers.")

# API Key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize conversation with a greeting
if 'first_run' not in globals():
    st.chat_message("assistant").write("Hello! How can I help you today?")
    first_run = False

# Chat input
user_input = st.chat_input("Type your question here...")

if user_input and api_key:
    # Display user message
    st.chat_message("user").write(user_input)
    
    with st.spinner("Searching for answers..."):
        try:
            # Initialize agent
            llm = ChatGroq(
                groq_api_key=api_key,
                model_name="Llama3-8b-8192",
                temperature=0.7
            )
            
            agent = initialize_agent(
                tools=[search, arxiv, wiki],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                verbose=False
            )
            
            # Get and display response
            response = agent.run({"input": user_input})
            st.chat_message("assistant").write(response)
            
        except Exception as e:
            st.error(f"Sorry, I encountered an error: {str(e)}")

elif user_input and not api_key:
    st.error("Please enter your Groq API key in the sidebar")
