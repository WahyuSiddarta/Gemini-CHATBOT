# Gemini Chatbot with FastAPI

This project is a simple HTTP server chatbot using Google's Gemini LLM and FastAPI.

## Features

- HTTP API endpoint for chat
- Integration with Gemini LLM (replace with your API key)

## Setup

1. Install dependencies:
   ```bash
   pip install fastapi uvicorn google-generativeai
   ```
2. Set your Gemini API key as an environment variable:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```
3. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

Send a POST request to `/chat` with JSON:

```json
{ "message": "Hello!" }
```

## Note

- Replace `your_api_key_here` with your actual Gemini API key.
