import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple
import google.generativeai as genai

app = FastAPI()

# Set up Gemini API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=gemini_api_key)


class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    message: Message




# Model routing constants
HARD_KEYWORDS = [
    r"prove", r"derivative", r"integral", r"induction", r"np[-\s]?hard", r"big-?o", r"complexity",
    r"dynamic programming", r"regex", r"sql", r"join", r"stack trace", r"panic", r"traceback", r"think hard"
]
MEDIUM_KEYWORDS = [
    r"explain", r"analyze", r"reasoning", r"difficult", r"challenge", r"debug", r"error", r"why"
]
HARD_PATTERN = re.compile("|".join(HARD_KEYWORDS), re.IGNORECASE)
MEDIUM_PATTERN = re.compile("|".join(MEDIUM_KEYWORDS), re.IGNORECASE)

def select_model(content: str) -> str:
    """Select Gemini model based on content complexity."""
    if HARD_PATTERN.search(content):
        return "gemini-pro"
    elif MEDIUM_PATTERN.search(content):
        return "gemini-flash"
    else:
        return "gemini-flash-lite"

# In-memory conversation storage: {(user_id, chat_id): [Message, ...]}
conversations: Dict[Tuple[str, str], List[Message]] = {}
# In-memory chat titles: {(user_id, chat_id): str}
chat_titles: Dict[Tuple[str, str], str] = {}


@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> Dict[str, str]:
    try:
        key = (request.user_id, request.chat_id)
        # Get or create conversation history
        history = conversations.setdefault(key, [])
        
        # Generate title for new conversation
        if key not in chat_titles and len(history) == 0 and request.message.role == "user":
            title_model = genai.GenerativeModel("gemini-2.5-flash-lite")
            title_prompt = f"Generate a short, descriptive title for this conversation: {request.message.content}"
            title_response = title_model.generate_content(title_prompt)
            chat_titles[key] = title_response.text.strip().replace("\n", " ")
        
        # Append the new message
        history.append(request.message)
        
        # Convert to Gemini format
        gemini_messages = [{"role": m.role, "parts": [m.content]} for m in history]

        # Select model based on content complexity
        model_name = select_model(request.message.content)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(gemini_messages)
        
        # Add assistant's reply to history
        history.append(Message(role="assistant", content=response.text))
        return {"response": response.text, "title": chat_titles.get(key), "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get all chat_ids and titles for a user
@app.get("/user/{user_id}/chats")
def get_chat_ids(user_id: str) -> Dict[str, List[Dict[str, str]]]:
    chat_list = [
        {"chat_id": chat_id, "title": chat_titles.get((uid, chat_id), "")}
        for (uid, chat_id) in conversations.keys() if uid == user_id
    ]
    return {"chats": chat_list}

# Get all messages and title for a user_id and chat_id
@app.get("/user/{user_id}/chat/{chat_id}")
def get_chat_history(user_id: str, chat_id: str) -> Dict[str, any]:
    key = (user_id, chat_id)
    history = conversations.get(key, [])
    messages = [{"role": m.role, "content": m.content} for m in history]
    title = chat_titles.get(key, "")
    return {"title": title, "messages": messages}
