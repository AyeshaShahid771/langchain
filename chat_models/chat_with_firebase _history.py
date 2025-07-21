from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env (API keys, etc.)
load_dotenv()
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env (API keys, etc.)
load_dotenv()
"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project
    - Copy the project ID
3. Create a Firestore database in the Firebase project
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""


# Firebase/Firestore Configuration
PROJECT_ID = "langchain-66976"
SESSION_ID = "user_session_new"  # Unique session ID per user or session
COLLECTION_NAME = "chat_history"

# Step 1: Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Step 2: Initialize Firestore-backed Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Step 3: Initialize Gemini Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    convert_system_message_to_human=True,  # required for memory use
)

# Step 4: Chat Loop
print("Start chatting with the Gemini AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # Store user's message in Firestore
    chat_history.add_user_message(human_input)

    # Get AI response with context from Firestore chat history
    ai_response = model.invoke(chat_history.messages)

    # Store AI's response
    chat_history.add_ai_message(ai_response.content)

    # Show AI response
    print(f"AI: {ai_response.content}")


# Firebase/Firestore Configuration
PROJECT_ID = "langchain-66976"
SESSION_ID = "user_session_new"  # Unique session ID per user or session
COLLECTION_NAME = "chat_history"

# Step 1: Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Step 2: Initialize Firestore-backed Message History
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Step 3: Initialize Gemini Chat Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    convert_system_message_to_human=True,  # required for memory use
)

# Step 4: Chat Loop
print("Start chatting with the Gemini AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    # Store user's message in Firestore
    chat_history.add_user_message(human_input)

    # Get AI response with context from Firestore chat history
    ai_response = model.invoke(chat_history.messages)

    # Store AI's response
    chat_history.add_ai_message(ai_response.content)

    # Show AI response
    print(f"AI: {ai_response.content}")
