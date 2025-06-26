# Dry Bean Genetics Research Chatbot

This project is a fullstack application designed to provide information about dry bean genetics, gene functions, and cultivar trial data through a conversational interface.

---

## Project Flow: How a Question is Answered

This section details the journey of a user's question from the moment it's typed in the frontend to the display of the answer.

1.  **Frontend: User Input**
    -   The user types a question into the input field in the React frontend (`frontend/src/App.jsx`).
    -   Pressing Enter or clicking the Send button triggers the `handleSend` function.

2.  **Frontend: Preparing the Request**
    -   The `handleSend` function captures the input text.
    -   It creates a user message object and adds it to the frontend's `messages` state, updating the UI to show the user's query.
    -   Crucially, it gathers the entire conversation history from the `messages` state using the `getConversationHistory` helper function. This history is formatted as a list of `{ role: 'user' | 'assistant', content: 'message text' }` objects.
    -   A POST request is initiated to the backend API endpoint `/api/chat` (defaulting to `http://localhost:8000/api/chat`).
    -   The request body includes the current `question` and the formatted `conversation_history`.
    -   The frontend sets a loading state and displays a loading indicator with rotating messages.

3.  **Backend: Receiving the Request**
    -   The FastAPI backend receives the POST request at the `/api/chat` endpoint (`backend/routes/chat.py`).
    -   The incoming JSON body is parsed and validated against the `ChatRequest` Pydantic model, which expects `question` (string) and `conversation_history` (optional list of dicts).
    -   The `chat_endpoint` function extracts the `question` and `conversation_history` from the validated request.

4.  **Backend: Processing the Question (`backend/services/pipeline.py`)**
    -   The `answer_question` function is called with the user's `question` and the `conversation_history`.
    -   The current user question is appended to the `conversation_history` list for use in subsequent API calls if needed.
    -   **Genetics Check:** The `is_genetics_question` function uses an OpenAI call (GPT-4o) to determine if the question is related to genetics or molecular biology. This is a simple true/false classification.
    -   **Non-Genetics Flow (e.g., Yield Data):**
        -   If the question is **not** genetic, the backend uses another OpenAI call (GPT-4o) with the `query_bean_data` function schema included.
        -   The model is prompted to decide if the user's question can be answered by calling this function, using the conversation history for context.
        -   **Function Call:** If the model decides to call `query_bean_data`, the arguments provided by the model are parsed.
        -   The `answer_bean_query` function (`backend/utils/bean_data.py`) is called with these arguments to query the loaded bean dataset (`Merged_Bean_Dataset.xlsx`). Logic is included to adjust the limit for 'highest yield' queries based on previous user feedback.
        -   `answer_bean_query` returns a markdown preview of the results and the full data markdown table.
        -   **Natural Language Summary:** If data is found, another OpenAI call (GPT-4o) is made. The prompt includes the original question, the conversation history, and the *preview* of the bean data results.
        -   The model is instructed to generate a natural language summary of the bean data, formatted according to specific markdown guidelines, directly addressing the user's question (including interpreting references to items in the table).
        -   The generated natural language answer, along with the full markdown table, is prepared for the final response.
    -   **Genetics Flow:**
        -   If the question *is* genetic, the backend proceeds with the RAG (Retrieval Augmented Generation) process.
        -   The question is embedded using two models: BGE (SentenceTransformer) and PubMedBERT.
        -   These embeddings are used to query two corresponding Pinecone indexes (`dry-bean-bge-abstract`, `dry-bean-pubmedbert-abstract`) to find relevant document summaries (abstracts/RAG text) based on similarity.
        -   The results from both models are combined and normalized based on their scores.
        -   The top relevant documents are retrieved using their DOIs (or source identifiers) from the pre-loaded RAG data (`summaries.jsonl`).
        -   The text summaries of these top documents are compiled as `context`.
        -   An OpenAI call (GPT-4o) is made using this `context`, the list of source DOIs, and the user's original question.
        -   The model is prompted to answer the question using *only* the provided context, formatted according to specific markdown guidelines.
        -   **Gene Extraction:** After the initial answer, another OpenAI call (GPT-4o) is made with a prompt to extract any gene or protein names mentioned in the answer text and provide a brief scientific summary for each, structured as a list of dictionaries.
        -   The generated answer text, the list of source DOIs, and the extracted gene information are prepared for the final response.

5.  **Backend: Constructing the Response**
    -   Based on whether the question was genetic or led to a bean data query, the backend gathers the generated `answer` (natural language summary or RAG answer), `sources` (list of DOIs or empty), `genes` (list of gene dicts or empty), and `full_markdown_table` (full table markdown or empty).
    -   This data is packaged into a `ChatResponse` Pydantic model.
    -   The backend sends this `ChatResponse` back to the frontend as a JSON payload.

6.  **Frontend: Displaying the Answer**
    -   The `handleSend` function in `App.jsx` receives the JSON response from the backend.
    -   It extracts the `answer`, `sources`, `genes`, and `full_markdown_table` from the response data.
    -   It constructs the final assistant message text by combining the main `answer`, formatted sources, formatted gene summaries, and the full markdown table (if present).
    -   This complete assistant message object is added to the `messages` state.
    -   React re-renders the UI, displaying the user's question and the assistant's response.
    -   The `chatEndRef` useEffect hook scrolls the chat area to show the latest message.
    -   The loading state is cleared.
    -   The `ReactMarkdown` component with the `remarkGfm` plugin renders the assistant's response, interpreting the markdown syntax (including tables, bold, italics, etc.).

---

## Key Technologies:

-   **Backend:** Python, FastAPI, Uvicorn, OpenAI API, Pinecone Client, Sentence Transformers, Hugging Face Transformers, Pandas, NumPy, orjson, python-multipart, python-dotenv
-   **Frontend:** React, Vite, Tailwind CSS, Axios, react-icons, react-markdown, remark-gfm

This flow describes the core interaction loop for answering user questions. Other parts of the application handle setup, environment variables, static file serving, etc.

## Features

- 🤖 Chat interface for research questions
- 📚 RAG-based context loading from research papers
- 🧬 Gene name extraction and NCBI mapping
- 📊 Bean trial data querying
- 📄 PDF/TXT file upload support
- 🌙 Dark/light theme support

## Tech Stack

### Backend
- FastAPI
- SentenceTransformers
- PubMedBERT
- Pinecone
- OpenAI GPT-4
- Pandas

### Frontend
- React + Vite
- Tailwind CSS
- React Markdown
- Axios

## Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Pinecone account
- OpenAI API key

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
GENE_DB_PATH=path/to/NCBI_Filtered_Data_Enriched.xlsx
RAG_FILE=path/to/summaries.jsonl
```

4. Run the backend:
```bash
uvicorn main:app --reload
```

### Frontend Setup

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Run the development server:
```bash
npm run dev
```

## Usage

1. Open `http://localhost:5173` in your browser
2. Type your research question in the chat input
3. View the response with sources and gene information
4. Upload PDF/TXT files for additional context

## API Endpoints

- `POST /api/chat` - Send a question and get a response
- `POST /api/upload` - Upload a PDF/TXT file
- `GET /api/ping` - Health check endpoint

## Development

### Backend Structure
```
backend/
├── main.py              # FastAPI app entry point
├── routes/             # API routes
│   ├── chat.py        # Chat endpoint
│   ├── upload.py      # File upload endpoint
│   └── ping.py        # Health check endpoint
├── services/          # Core business logic
│   └── pipeline.py    # Main processing pipeline
└── utils/             # Utility functions
    ├── ncbi_utils.py  # NCBI data handling
    └── bean_data.py   # Bean trial data
```

### Frontend Structure
```
frontend/
├── src/
│   ├── components/    # React components
│   │   ├── ChatBox.jsx
│   │   └── FileUpload.jsx
│   ├── App.jsx       # Main app component
│   └── index.css     # Styles
└── package.json      # Dependencies
```

## License

MIT 