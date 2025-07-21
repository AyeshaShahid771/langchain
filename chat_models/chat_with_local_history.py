from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    convert_system_message_to_human=True,  # Required for SystemMessage compatibility
)

chat_history = []

# Optional system message (converted to HumanMessage internally)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    # Invoke model with full history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI: {response}")

# Display the message history
print("---- Message History ----")
for message in chat_history:
    print(message)
