Document Intelligence (RAG) project

Files:
- app.py              : Streamlit app (main demo)
- Java-Document-RAG.ipynb : notebook version
- Java_Programming_Notes.pdf : sample document
- requirements.txt    : dependencies

Run:
1. pip install -r requirements.txt
2. streamlit run app.py
3. In the sidebar, select an AI provider and enter its API key, then upload a PDF,
   DOCX, TXT, or Markdown file and create a knowledge base.
4. Ask questions in the conversation area. Answers use the uploaded document as
   context and include source sections where available.

The responsive Streamlit interface keeps provider settings and document upload in
the sidebar, with document status, conversation, and source details in the main
workspace.

API keys can be entered in the UI or read from the matching environment variable:
GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY. Do not commit
API keys to source control; rotate any key that has been exposed.
