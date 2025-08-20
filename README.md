# 🤖 Intelligent Gemini Chatbot with Model Routing

> A simple FastAPI-based chatbot that intelligently routes conversations to different Gemini models based on query complexity and conversation context.

## ✨ Features

### 🎯 **Smart Model Routing**

- **Dynamic Model Selection**: Automatically chooses between `gemini-2.5-pro`, `gemini-2.5-flash`, or `gemini-2.5-flash-lite` based on:
  - Query complexity analysis
  - Keyword detection (technical terms, mathematical concepts)
  - Conversation token count
  - Question complexity scoring

### 💬 **Chat Management**

- **Multi-User Support**: Separate conversation histories per user and chat session
- **Conversation Limits**: Maximum 100 messages per chat to maintain performance
- **Context Awareness**: Maintains last 20 messages for efficient context handling
- **Auto-Generated Titles**: Smart, concise conversation titles (max 5 words)

### 🔧 **Production-Ready Features**

- **Environment Variables**: Secure API key management with `.env` support
- **Error Handling**: Comprehensive exception handling with meaningful error messages
- **RESTful API**: Clean endpoints for chat, history retrieval, and user management
- **Token Optimization**: Intelligent token counting for cost-effective API usage

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd final-project

# Install dependencies
pip install fastapi uvicorn google-generativeai python-dotenv
```

### 2. Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --port 8001

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8001
```

## 📚 API Endpoints

### 💬 **Chat Endpoint**

```http
POST /chat
Content-Type: application/json

{
  "user_id": "user123",
  "chat_id": "chat456",
  "message": {
    "role": "user",
    "content": "Explain quantum computing"
  }
}
```

**Response:**

```json
{
  "response": {
    "role": "assistant",
    "content": "Quantum computing is..."
  },
  "title": "Quantum Computing Explanation",
  "model": "gemini-2.5-flash"
}
```

### 📋 **Get User Chats**

```http
GET /user/{user_id}/chats
```

### 📖 **Get Chat History**

```http
GET /user/{user_id}/chat/{chat_id}
```

## 🧠 Model Routing Logic

The system intelligently selects the appropriate Gemini model based on:

| Condition                          | Model Used              | Use Case                                            |
| ---------------------------------- | ----------------------- | --------------------------------------------------- |
| Complex queries + High token count | `gemini-2.5-pro`        | Advanced technical discussions, mathematical proofs |
| Medium complexity                  | `gemini-2.5-flash`      | General explanations, moderate complexity           |
| Simple queries                     | `gemini-2.5-flash-lite` | Basic questions, casual conversation                |

**Complexity Scoring:**

- **Hard Keywords** (+5 each): `derivative`, `integral`, `algorithm`, `complexity`, etc.
- **Medium Keywords** (+2 each): `explain`, `analyze`, `why`, `how`, etc.
- **Question Marks** (+1 each): Indicates inquiry complexity

## 🛠️ Architecture

```
├── main.py              # FastAPI application with routing logic
├── .env                 # Environment variables (API keys)
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

## 🔒 Security Features

- ✅ API keys stored in environment variables
- ✅ Input validation with Pydantic models
- ✅ Comprehensive error handling
- ✅ Conversation limits to prevent abuse

## 🚧 Production Considerations

Current implementation uses in-memory storage. For production deployment, consider:

1. **Database Integration**: Replace in-memory dictionaries with Redis/PostgreSQL
2. **Model Distillation**: Use higher models to improve lower model responses
3. **Fact Storage**: Implement knowledge base for consistency
4. **Query Transformation**: Break complex questions into simpler parts
5. **Rate Limiting**: Implement API rate limiting
6. **Authentication**: Add user authentication and authorization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License.

---

**⚡ Built with FastAPI + Google Gemini AI for intelligent, cost-effective conversational AI**
