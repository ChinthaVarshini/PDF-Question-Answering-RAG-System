# PDF Question Answering with RAG

Ask questions about your documents and get answers grounded in their contents. This project is a Streamlit application that builds a searchable knowledge base from an uploaded file, retrieves relevant passages, and uses an AI model to answer questions with source references.

## Features

- Upload PDF, DOCX, TXT, or Markdown documents.
- Create a vector index with Hugging Face sentence-transformer embeddings and FAISS.
- Ask questions in natural language using a choice of AI providers.
- Retrieve relevant document passages for each question and show their source pages or sections.
- Configure the model and provide an API key in the sidebar or through an environment variable.
- Clear the conversation without rebuilding the knowledge base.

## Requirements

- Python 3
- An API key for one of the supported AI providers

## Getting started

1. Clone the repository and move into the project directory:

   ```bash
   git clone https://github.com/ChinthaVarshini/PDF-Question-Answering-RAG-System.git
   cd PDF-Question-Answering-RAG-System
   ```

2. Create and activate a virtual environment:

   **Windows (PowerShell)**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies and launch the app:

   ```bash
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

4. In the app, choose a provider, enter its API key, upload a document, and select **Create knowledge base**. Once indexing is complete, ask questions in the conversation area.

The sentence-transformer embedding model is downloaded when it is first needed.

## Supported AI providers

| Provider | Environment variable |
| --- | --- |
| Groq | `GROQ_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |

Set the environment variable for your chosen provider before starting Streamlit, or enter the key in the app's sidebar. The model name can also be changed in the sidebar.

## How it works

1. The application extracts text from the uploaded document.
2. Text is split into overlapping chunks and converted into embeddings.
3. FAISS indexes the embeddings for similarity search.
4. For each question, the most relevant chunks are sent to the selected language model as context.
5. The response is generated from that context, with retrieved passages available under **Sources**.

## Project files

- `app.py` — Streamlit application and retrieval-augmented question-answering flow.
- `Java-Document-RAG.ipynb` — notebook version of the project.
- `Java_Programming_Notes.pdf` — sample document for trying the application.
- `requirements.txt` — Python dependencies.

## Security

Never commit API keys or other credentials. Environment variables and keys entered in the sidebar are intended to keep credentials out of source files. If a key is accidentally exposed, revoke it with the provider and create a replacement.
