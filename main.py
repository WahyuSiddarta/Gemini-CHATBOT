import os
from dotenv import load_dotenv
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any
from google import genai
from google.genai import types


# Load environment variables from .env file
load_dotenv()
app = FastAPI()

# Set up Gemini API key from environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=gemini_api_key)


class Message(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    message: Message




# Model routing constants
HARD_KEYWORDS = [
    r"derivative", r"integral",  r"big-?o", r"complexity",
    r"dynamic programming", r"regex", r"sql",  r"stack trace", r"panic", r"traceback",  r"recursion", r"algorithm", r"theorem", r"bukti", r"turunan", r"integral", r"induksi",
    r"np[-\s]?sulit", r"kompleksitas", r"pemrograman dinamis", r"jejak tumpukan", r"jejak kesalahan", r"jejak error", r"jejak",
    r"algoritma", r"teorema", r"persamaan", r"matematika", r"logika", r"berpikir keras", r"pikir keras", r"buktikan", r"soal sulit",
    r"tantangan", r"uji", r"uji coba", r"uji hipotesis"
]
MEDIUM_KEYWORDS = [
    r"apa itu", r"jelaskan", r"analisa", r"penjelasan", r"mengapa", r"kenapa", r"sulit", r"tantangan", r"perbaiki", r"kesalahan",
    r"masalah", r"solusi", r"langkah", r"cara", r"bagaimana", r"penyebab", r"penyelesaian"
]
HARD_PATTERN = re.compile("|".join(HARD_KEYWORDS), re.IGNORECASE)
MEDIUM_PATTERN = re.compile("|".join(MEDIUM_KEYWORDS), re.IGNORECASE)

def select_model(content: str) -> str:
    """Select Gemini model based on weighted content complexity."""
    score = 0
    # Hard keywords: +5 each
    score += 5 * len(HARD_PATTERN.findall(content))
    # Medium keywords: +2 each
    score += 2 * len(MEDIUM_PATTERN.findall(content))
    # Question marks: +1 each
    score += content.count('?')
    if score >= 10:
        return "gemini-2.5-pro"
    elif score >= 5:
        return "gemini-2.5-flash"
    else:
        return "gemini-2.5-flash-lite"

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
            title_prompt = f"Generate a single, short, and concise title (max 5 words, no explanation) for this conversation: {request.message.content}"
            title_response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=title_prompt
            )
            # Sanitize: take only the first line, remove bullet/numbering, and trim to 5 words max
            raw_title = title_response.candidates[0].content.parts[0].text.strip()
            first_line = raw_title.split('\n')[0]
            # Remove bullet points or numbering
            first_line = first_line.lstrip('-*0123456789. ').strip()
            # Limit to 5 words
            words = first_line.split()
            concise_title = ' '.join(words[:5])
            chat_titles[key] = concise_title
        
        # Append the new message
        history.append(request.message)

        # Prepare prompt for Gemini API (latest format expects a single string)
        prompt = "You are a helpful assistant. Respond to the user's message in a clear and concise manner.\n"
        for m in history:
            prompt += f"{m.role}: {m.content}\n"

        # Select model based on content complexity
        model_name = select_model(request.message.content)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        # Add assistant's reply to history
        assistant_reply = response.candidates[0].content.parts[0].text
        history.append(Message(role="assistant", content=assistant_reply))
        return {"response": assistant_reply, "title": chat_titles.get(key), "model": model_name}
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
def get_chat_history(user_id: str, chat_id: str) -> Dict[str, Any]:
    key = (user_id, chat_id)
    history = conversations.get(key, [])
    messages = [{"role": m.role, "content": m.content} for m in history]
    title = chat_titles.get(key, "")
    return {"title": title, "messages": messages}

# running /Users/wahyu/Code/H8/data-scientist/final-project/.venv/bin/python -m uvicorn main:app --reload --port 8001