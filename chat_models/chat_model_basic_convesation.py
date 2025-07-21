from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model with system message conversion
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20", convert_system_message_to_human=True
)

# First message sequence
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the Gemini model
result = model.invoke(messages)
print(f"Answer from Gemini AI: {result.content}")

# Second message sequence with context
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke again
result = model.invoke(messages)
print(f"Answer from Gemini AI: {result.content}")
