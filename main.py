import os
from dotenv import load_dotenv
import re
from fastapi import FastAPI, HTTPException
import concurrent.futures
from pydantic import BaseModel
from typing import Dict, List, Tuple, Any
from google import genai
from google.genai import types
import re as _re
# Load environment variables from .env file
load_dotenv()

# === Constants ===
# HTTP STATUS
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_INTERNAL_ERROR = 500

# CHAT LIMIT
MAX_CHAT_HISTORY = 20  # Number of messages to keep in context
MAX_TOTAL_CONVERSATION = 100  # Max messages per chat
MODEL_TOP_TIER_THRESHOLD = 10
MODEL_MEDIUM_TIER_THRESHOLD = 7

# MESSAGE LIMITS
TOKEN_THRESHOLD_PRO = 3000
WEIGHT_SMALL = 2
WEIGHT_MEDIUM = 3
WEIGHT_HARD = 5

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
    r"derivative", r"integral",  r"big-?o", r"complexity", r"algorithm", r"algoritma",r"mathematical",
    r"dynamic programming", r"regex", r"sql",  r"stack trace", r"panic", r"traceback",  r"recursion", r"algorithm", r"theorem", r"bukti", r"turunan", r"integral", r"induksi",
    r"np[-\s]?sulit", r"kompleksitas", r"pemrograman dinamis", r"jejak tumpukan", r"jejak kesalahan", r"jejak error", r"jejak",
    r"algoritma", r"teorema", r"persamaan", r"matematika", r"logika", r"berpikir keras", r"pikir keras", r"buktikan", r"soal sulit",
    r"tantangan", r"uji", r"uji coba", r"uji hipotesis", r"prima", r"prime"
]
MEDIUM_KEYWORDS = [
    r"apa itu", r"jelaskan", r"analisa", r"penjelasan", r"mengapa", r"kenapa", r"sulit", r"tantangan", r"perbaiki", r"kesalahan",
    r"masalah", r"solusi", r"langkah", r"cara", r"bagaimana", r"penyebab", r"penyelesaian" 
]
HARD_PATTERN = re.compile("|".join(HARD_KEYWORDS), re.IGNORECASE)
MEDIUM_PATTERN = re.compile("|".join(MEDIUM_KEYWORDS), re.IGNORECASE)

# simple model routing, but current method is not suitable for production
# context on previous message maybe lost, potential solution for production
# 1. distilation using higher model
# 2. store message fact to make lower model smarter, also consistency check for lower model if answer is same as fact
# 3. transform question into statement or smaller question to make lower model easier to process
def select_model(content: str, conversation: str) -> str:
    """Select Gemini model based on weighted content complexity."""
    score = 1  # Start with a base score of 1
    # Hard keywords: +5 each
    score += WEIGHT_HARD * len(HARD_PATTERN.findall(content))
    # Medium keywords: +3 each
    score += WEIGHT_MEDIUM * len(MEDIUM_PATTERN.findall(content))
    # Question marks: +2 each
    score += WEIGHT_SMALL * content.count('?')
    
    # Convert conversation (list of Message) to a string prompt for token counting
    # weighting for pro model is disabled because of potential token limit issues
    # sometimes, query get {'error': {'code': 500, 'message': 'An internal error has occurred. Please retry or report in https://developers.generativeai.google/guide/troubleshooting', 'status': 'INTERNAL'}} 
    # if isinstance(conversation, list):
    #     prompt = ""
    #     for m in conversation:
    #         prompt += f"{m.role}: {m.content}\n"
    #     estimatedToken = client.models.count_tokens(model="gemini-2.5-flash-lite", contents=prompt)
    # else:
    #     estimatedToken = client.models.count_tokens(conversation)
    # token_count = estimatedToken.total_tokens if hasattr(estimatedToken, 'total_tokens') else int(estimatedToken)
    
    # scoring base model routing
    # if score >= MODEL_TOP_TIER_THRESHOLD or token_count > TOKEN_THRESHOLD_PRO:
    #     return "gemini-2.5-pro"
    # elif score >= MODEL_MEDIUM_TIER_THRESHOLD:
    #     return "gemini-2.5-flash"
    # else:
    #     return "gemini-2.5-flash-lite"
    if score >= MODEL_MEDIUM_TIER_THRESHOLD:
        return "gemini-2.5-flash"
    else:
        return "gemini-2.5-flash-lite"

# In-memory conversation storage: {(user_id, chat_id): [Message, ...]}
conversations: Dict[Tuple[str, str], List[Message]] = {}
# In-memory chat titles: {(user_id, chat_id): str}
chat_titles: Dict[Tuple[str, str], str] = {}

def wrap_code_blocks(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Handle triple backtick code blocks (with or without language)
    text = _re.sub(r"```([a-zA-Z0-9]*)\n([\s\S]*?)```", lambda m: f"{m.group(2).strip()}", text)
    # Handle indented code blocks (4 spaces or tab)
    lines = text.split('\n')
    in_code = False
    code_lines = []
    result_lines = []
    for line in lines:
        if (line.startswith('    ') or line.startswith('\t')):
            code_lines.append(line.lstrip())
            in_code = True
        else:
            if in_code:
                result_lines.append(f"{'\n'.join(code_lines)}")
                code_lines = []
                in_code = False
            result_lines.append(line)
    if in_code:
        result_lines.append(f"{'\n'.join(code_lines)}")
    return '\n'.join(result_lines)
        
def add_citations(response):
    text = ""
    grounding_metadata = getattr(response.candidates[0], "grounding_metadata", None)
    if not grounding_metadata:
        return text
    supports = getattr(grounding_metadata, "grounding_supports", None)
    chunks = getattr(grounding_metadata, "grounding_chunks", None)
    if not supports or not chunks:
        return text

    # Collect all unique citations (index, url)
    citation_map = {}
    for i, chunk in enumerate(chunks):
        if getattr(chunk, "web", None) and getattr(chunk.web, "uri", None):
            citation_map[i + 1] = chunk.web.uri

    # Build bibliography at the end
    if citation_map:
        bib = "\n\n Referensi:\n"
        for idx, url in sorted(citation_map.items()):
            bib += f"[{idx}] {url}\n"
        return  bib.strip()
    else:
        return ""

@app.post("/chat")
def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    def process_chat():
        key = (request.user_id, request.chat_id)
        # Get or create conversation history
        history = conversations.setdefault(key, [])
        # Enforce total conversation limit
        if len(history) >= MAX_TOTAL_CONVERSATION:
            return {
                "response": "Conversation limit is ended. Please start a new chat.",
                "title": chat_titles.get(key),
                "model": None
            }

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
        # Limit to the last MAX_CHAT_HISTORY messages (10 back-and-forth) for quicker and limit token input
        limited_history = history[-MAX_CHAT_HISTORY:]
        prompt = ""
        for m in limited_history:
            prompt += f"{m.role}: {m.content}\n"
       
        config = types.GenerateContentConfig(
            tools=[
                types.Tool(code_execution=types.ToolCodeExecution),
                types.Tool(google_search=types.GoogleSearch())
            ],
            system_instruction="You are a helpful assistant. Use Google Search if needed to ground your answers and cite sources with [number] where relevant. Use Code execution tool only for code-related queries and complex math, do not show internal tool in response if it being used"
        )

        # Select model based on content complexity
        model_name = select_model(request.message.content, history)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        # Add assistant's reply to history
        # Safely extract text from response, handling different response structures
        assistant_reply = ""
        content_parts = []
        debug = ""
        if response.candidates and len(response.candidates) > 0:
            debug += "1"
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                debug += "2"
                parts = candidate.content.parts
                if parts and len(parts) > 0:
                    debug += "3"
                    for part in parts:
                        # Always keep natural language
                        if getattr(part, "text", None):
                            content_parts.append(part.text)

                        # Show code only if Gemini intended it as part of the answer
                        if getattr(part, "executable_code", None):
                            if candidate.finish_reason == "STOP":  # user-facing
                                content_parts.append(f"{part.executable_code.code}")
                            # else: skip (tool-only, like search)

                        # Show code execution result only if it’s relevant
                        if getattr(part, "code_execution_result", None):
                            if part.code_execution_result.output:
                                if candidate.finish_reason == "STOP":
                                    content_parts.append(
                                        f"Output: {part.code_execution_result.output}"
                                    )
                                # else skip (internal tool use)

        assistant_reply = "".join(content_parts)
        
        if not assistant_reply:
            assistant_reply = "I apologize, but I couldn't generate a proper response. Please try again.".join(debug) 

        # Wrap all code blocks in <code>...</code> tags
        
        

        assistant_reply = wrap_code_blocks(assistant_reply)
        history.append(Message(role="assistant", content=assistant_reply))
        return {
            "response": {"role": "assistant", "content": assistant_reply},
            "title": chat_titles.get(key),
            "model": model_name,
            "citation": add_citations(response)
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_chat)
            try:
                result = future.result(timeout=25)
                return result
            except concurrent.futures.TimeoutError:
                return {
                    "response": "Query failed: process exceeded 25 seconds.",
                    "title": None,
                    "model": None
                }
    except Exception as e:
        raise HTTPException(status_code=HTTP_STATUS_INTERNAL_ERROR, detail=str(e))


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

# Delete a specific chat conversation and title
@app.delete("/user/{user_id}/chat/{chat_id}")
def delete_chat(user_id: str, chat_id: str) -> Dict[str, str]:
    """Delete a specific chat conversation and its title from memory."""
    key = (user_id, chat_id)
    
    # Check if chat exists
    if key not in conversations:
        raise HTTPException(status_code=HTTP_STATUS_NOT_FOUND, detail="Chat not found")
    
    # Remove conversation history
    del conversations[key]
    
    # Remove chat title if it exists
    if key in chat_titles:
        del chat_titles[key]
    
    return {"message": f"Chat {chat_id} for user {user_id} has been deleted successfully"}

# running /Users/wahyu/Code/H8/data-scientist/final-project/.venv/bin/python -m uvicorn main:app --reload --port 8001