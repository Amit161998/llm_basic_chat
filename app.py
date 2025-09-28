# Import necessary libraries
import streamlit as st  # For building the web app interface
import os  # For environment variable management
import openai  # For OpenAI API access
from langchain_openai import ChatOpenAI  # For using OpenAI LLM with LangChain
from langchain_core.output_parsers import StrOutputParser  # For parsing LLM output
# For creating prompt templates
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Set up Langsmith tracking environment variables
os.environ['LANGSMITH_API_KEY'] = os.getenv(
    'LANGSMITH_API_KEY')  # API key for LangChain
os.environ['LANGSMITH_TRACING'] = "true"  # Enable LangChain tracing
# Set project name for tracking
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

# Define the prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        # System message to guide the assistant's behavior
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        # User message template with a placeholder for the question
        ("user", "Question: {question}")
    ]
)


def generate_response(question, api_key, llm, temperature, max_tokens):
    """
    Generates a response from the OpenAI language model using the provided parameters.
    Args:
        question (str): The user's question.
        api_key (str): OpenAI API key.
        llm (str): The model name to use (e.g., 'gpt-4').
        temperature (float): Sampling temperature for response randomness.
        max_tokens (int): Maximum number of tokens in the response.
    Returns:
        str: The generated answer from the model.
    """
    openai.api_key = api_key  # Set the OpenAI API key
    # Initialize the language model with the selected model name
    llm = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    # Set up the output parser to extract string responses
    output_parser = StrOutputParser()
    # Create a processing chain: prompt -> LLM -> output parser
    chain = prompt | llm | output_parser
    # Invoke the chain with the user's question
    answer = chain.invoke({'question': question})
    return answer  # Return the generated answer


# Set the title of the Streamlit app
st.title("Enhanced Q&A Chatbot with OpenAI")

# Sidebar settings for user input and configuration
st.sidebar.title("Settings")  # Sidebar title
api_key = st.sidebar.text_input(
    "Enter your Open AI API Key:", type="password")  # Input for API key

# Dropdown to select different OpenAI models
llm = st.sidebar.selectbox("Select an Open AI Model", [
                           "gpt-4o", "gpt-4-turbo", "gpt-4"])  # Model selection

# Sliders to adjust response parameters
temperature = st.sidebar.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=0.7)  # Adjust randomness
max_tokens = st.sidebar.slider(
    "Max Tokens", min_value=50, max_value=300, value=150)  # Adjust response length

# Main interface for user input
st.write("Go Ahead and ask any question")  # Instruction for the user
user_input = st.text_input("You:")  # Text input for user's question

# If user provides input, generate and display the response
if user_input:
    response = generate_response(
        user_input, api_key, llm, temperature, max_tokens)  # Get response from model
    st.write(response)  # Display the response
else:
    # Prompt user to enter input
    st.write("Please provide some user input to proceed with response")
