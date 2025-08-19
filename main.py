import os
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


from typing import List


class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    message: Message




# In-memory conversation storage: {(user_id, chat_id): [Message, ...]}
conversations: Dict[Tuple[str, str], List[Message]] = {}
# In-memory chat titles: {(user_id, chat_id): str}
chat_titles: Dict[Tuple[str, str], str] = {}


@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> Dict[str, str]:
    try:
        import re
        key = (request.user_id, request.chat_id)
        # Get or create conversation history
        history = conversations.setdefault(key, [])
        # If this is the first user message, generate a title
        if key not in chat_titles and len(history) == 0 and request.message.role == "user":
            title_model = genai.GenerativeModel("gemini-2.5-flash-lite")
            title_prompt = f"Generate a short, descriptive title for this conversation: {request.message.content}"
            title_response = title_model.generate_content(title_prompt)
            chat_titles[key] = title_response.text.strip().replace("\n", " ")
        # Append the new message
        history.append(request.message)
        # Convert to Gemini format
        gemini_messages = [
            {"role": m.role, "parts": [m.content]} for m in history
        ]

        # Weighted model routing based on keywords
        user_content = request.message.content.lower()
        score = 0
        # High-weight (hard) keywords
        hard_keywords = [
            r"prove", r"derivative", r"integral", r"induction", r"np[-\s]?hard", r"big-?o", r"complexity",
            r"dynamic programming", r"regex", r"sql", r"join", r"stack trace", r"panic", r"traceback", r"think hard"
        ]
        # Medium-weight keywords
        medium_keywords = [
            r"explain", r"analyze", r"reasoning", r"difficult", r"challenge", r"debug", r"error", r"why"
        ]
        for kw in hard_keywords:
            if re.search(kw, user_content):
                score += 10
        for kw in medium_keywords:
            if re.search(kw, user_content):
                score += 3

        if score >= 10:
            model_name = "gemini-pro"
        elif score >= 3:
            model_name = "gemini-flash"
        else:
            model_name = "gemini-flash-lite"

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(gemini_messages)
        # Add assistant's reply to history
        history.append(Message(role="assistant", content=response.text))
        return {"response": response.text, "title": chat_titles.get(key), "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Get all chat_ids and titles for a user
@app.get("/user/{user_id}/chats")
def get_chat_ids(user_id: str) -> Dict[str, list]:
    chat_list = []
    for (uid, chat_id) in conversations.keys():
        if uid == user_id:
            chat_list.append({
                "chat_id": chat_id,
                "title": chat_titles.get((uid, chat_id), "")
            })
    return {"chats": chat_list}

# Get all messages and title for a user_id and chat_id
@app.get("/user/{user_id}/chat/{chat_id}")
def get_chat_history(user_id: str, chat_id: str) -> Dict[str, list]:
    key = (user_id, chat_id)
    history = conversations.get(key, [])
    # Return as list of dicts
    messages = [{"role": m.role, "content": m.content} for m in history]
    title = chat_titles.get(key, "")
    return {"title": title, "messages": messages}
