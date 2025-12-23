import streamlit as st
from pathlib import Path
import os
import sqlite3
from sqlalchemy import create_engine
from dotenv import load_dotenv

# MODERN LANGCHAIN COMMUNITY IMPORTS
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

load_dotenv()
os_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLLite 3 Database- Student.db", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB", options=radio_opt)

# Database Input Logic
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Host")
    mysql_user = st.sidebar.text_input("User")
    mysql_password = st.sidebar.text_input("Password", type="password")
    mysql_db = st.sidebar.text_input("Database")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Groq API Key", type="password", value=os_api_key if os_api_key else "")

if not api_key:
    st.info("Please add the Groq API key to continue.")
    st.stop()

# LLM model - Temperature 0 is best for SQL generation accuracy
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", temperature=0, streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, h=None, u=None, p=None, d=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        if not dbfilepath.exists():
            st.error(f"Database file not found. Run your sqlite.py script first.")
            st.stop()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{u}:{p}@{h}/{d}"))

db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# --- THE FIX FOR AGENT_SCRATCHPAD ERROR ---
# Using agent_type="tool-calling" ensures the prompt includes 'agent_scratchpad' correctly.
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling", # Modern standard for Llama 3.1
    max_iterations=30,          # Prevents the "Iteration Limit" error
    handle_parsing_errors=True  # Safely handles any formatting slips from the LLM
)

# Chat Session Logic
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        container = st.container()
        streamlit_callback = StreamlitCallbackHandler(container)
        
        # agent.run handles the input and output strings automatically
        response = agent.run(user_query, callbacks=[streamlit_callback])
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)