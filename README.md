# Nha Trang Travel Chatbot
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/huylys54/NhaTrangChatbot)


A conversational AI assistant for travel information about Nha Trang, Vietnam. This project combines retrieval-augmented generation (RAG), hybrid search (BM25 + embeddings), and streaming chat UI.

## Features

- Conversational chatbot for Nha Trang tourism
- Hybrid document retrieval (BM25 + Chroma vector search)
- Real-time web search and Google Maps integration
- Streaming responses via FastAPI backend
- Basic modern Streamlit-based chat UI

## Project Structure

```
.
├── chat_ui.py                # Streamlit UI for chat
├── indexing_docs.py          # Indexing markdown documents
├── config.py                 # Configuration and constants
├── src/
│   ├── api/                  # FastAPI backend
│   └── rag/                  # RAG, agent, embedder, chunker
├── data/                     # Data and vector stores
├── tests/                    # Unit tests
└── requirements.txt          # Python dependencies
```

## Getting Started

### 0. (Recommended) Create and activate a virtual environment

```sh
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 0.5. Setup environment variables
Copy the example environment file and edit as needed:
```sh
cp .env.example .env
```
Edit `.env` to set your API keys.

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Prepare data

- Place your markdown documents in `data/raw/`
- Run the document chunker to preprocess and split documents

### 3. Build the index

You can build or update the hybrid index using the provided script:

```sh
python indexing_docs.py
```

You can customize the chunk size, overlap, and directories using command-line arguments.
For example:

```sh
python indexing_docs.py --dir_path data/raw --persist_directory data/chroma --bm25_directory data/bm25
```

### 4. Start the API server

```sh
uvicorn src.api.main:app --reload
```

### 5. Launch the chat UI

```sh
streamlit run chat_ui.py
```

## Configuration

See [`config.py`](config.py) for directory paths, model names, and other settings.

## API Endpoints

- `POST /chat` — Stream chat responses
- `GET /history` — Get chat history
- `DELETE /history` — Clear chat history

## Concepts

### Retrieval-Augmented Generation (RAG)

RAG is a technique that combines information retrieval with generative AI. Instead of relying solely on the language model's internal knowledge, it retrieves relevant documents from an external knowledge base and uses them as context for the model to generate more accurate, up-to-date, and grounded answers.


### Hybrid Search

In this project, I combine **BM25 search** and **embedding-based vector search** to retrieve relevant documents. The combined score for each document is calculated with:

```
bm25_weight * (1 / rank_bm25) + emb_weight * (1 / rank_emb)
```

* **BM25 Search**: A traditional keyword-based retrieval algorithm that ranks documents based on term frequency and inverse document frequency. It works well for exact and sparse keyword matches.
* **Embedding (Vector) Search**: Converts text into dense vector representations using embeddings. It retrieves documents based on semantic similarity using vector distance metrics.

This hybrid approach balances keyword precision and semantic relevance.


### RAG Pipeline Implementation

The conversational assistant follows this flow:

1. **User Query**

2. **Detect Language**

3. **Classify Intention**: Decide whether the query aims to:

   * **Retrieve travel info**
   * **Search general web info**
   * **Get a location**

4. **Execute Tool Based on Intention**:

   * **Retrieve**: Translate query to Vietnamese, then retrieve documents using hybrid search.
   * **Search**: Transform query for better web search, then use Tavily API.
   * **Get Location**: Transform query to extract location info, then query Google Maps API.

5. **Error Handling & Fallback**:

   * If **retrieve** or **get location** fails → fallback to **search**
   * If **search** fails → raise error and stop.

6. **Generate Final Answer**: Use retrieved or searched context to generate a response.

<p align="center">
  <img src="./state_graph.png">
</p>

### Streaming Response vs Full Response Graph

Implemented two graph variations using **LangGraph**:

- **Full Response Graph**:
  Conditional edge for error handling before final generation:

  ```python
  graph.add_conditional_edges(
      "handle_error",
      lambda s: "generate_response" if not s["error"] or s["intent"] == "search" else "execute_tool",
      {"generate_response": "generate_response", "execute_tool": "execute_tool"}
  )
  ```

- **Streaming Response Graph**:
  Skip LLM generation logic inside the graph. Terminate after error handling — streaming handled separately outside:

  ```python
  graph.add_edge("handle_error", END)
  ```


### Memory & Conversation History

The assistant maintains short-term memory using:

**ConversationBufferWindowMemory (k=5)**
This keeps the last 5 interactions for context continuity within the session.



## Customization

- To add new data, place markdown files in your chosen `--dir-path` (default: `data/raw/`) and re-run [`indexing_docs.py`](indexing_dodcs.py).
- To change chunking or indexing parameters, edit the arguments in [`indexing_docs.py`](indexing_dodcs.py) or pass them via the command line.
- To change models, update settings in [`config.py`](config.py).

## Contributing

Pull requests and issues are welcome!

## License

This project is licensed under the MIT License.
