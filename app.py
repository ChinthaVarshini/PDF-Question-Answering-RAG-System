import os, tempfile
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="Document Intelligence | Ask your documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17233b;
        --muted: #718096;
        --accent: #5668e8;
        --line: #e7ebf3;
        --surface: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(86, 104, 232, 0.08), transparent 34%),
            #f7f8fc;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    .block-container {
        max-width: 1180px;
        padding: 3rem 2.5rem 4rem;
    }
    h1, h2, h3, [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        letter-spacing: -0.035em;
    }
    h1 { font-size: clamp(2rem, 4vw, 3rem) !important; font-weight: 800 !important; }
    h2 { font-weight: 700 !important; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }

    [data-testid="stSidebar"] {
        background: #111a2e;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #eef2ff;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #a4aec5; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.05);
        border: 1px dashed rgba(207,216,255,0.38);
        border-radius: 12px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #e4e9f8; }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        color: #eef2ff;
    }
    [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
        border-color: rgba(255,255,255,0.35);
        color: #ffffff;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 22px;
    }
    .brand-icon {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: linear-gradient(145deg, #7888ff, #5263df);
        box-shadow: 0 8px 22px rgba(86,104,232,0.3);
        font-size: 20px;
    }
    .brand-name {
        color: #ffffff;
        font: 700 16px 'Manrope', sans-serif;
        letter-spacing: -0.03em;
    }
    .brand-subtitle {
        margin-top: 2px;
        color: #9da9c3;
        font: 500 11px 'DM Sans', sans-serif;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    .section-label {
        margin: 20px 0 8px;
        color: #aab5cd;
        font: 700 11px 'DM Sans', sans-serif;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .eyebrow {
        margin-bottom: 10px;
        color: var(--accent);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
    }
    .hero-copy {
        max-width: 680px;
        margin-top: -8px;
        font-size: 16px;
        line-height: 1.7;
    }
    .welcome-card {
        margin-top: 28px;
        padding: 34px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f5f6ff 100%);
        box-shadow: 0 12px 35px rgba(33, 48, 92, 0.05);
    }
    .welcome-icon {
        display: grid;
        place-items: center;
        width: 52px;
        height: 52px;
        margin-bottom: 20px;
        border-radius: 16px;
        background: #eef0ff;
        font-size: 25px;
    }
    .welcome-title {
        margin-bottom: 8px;
        color: var(--ink);
        font: 700 20px 'Manrope', sans-serif;
        letter-spacing: -0.03em;
    }
    .welcome-text {
        max-width: 640px;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
    }
    .welcome-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 24px;
    }
    .welcome-step {
        padding: 14px;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: rgba(255,255,255,0.8);
        color: #526078;
        font-size: 12px;
        line-height: 1.5;
    }
    .welcome-step strong {
        display: block;
        margin-bottom: 5px;
        color: var(--ink);
        font-size: 13px;
    }
    [data-testid="stMetric"] {
        padding: 16px 18px;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: 0 5px 18px rgba(33, 48, 92, 0.035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 5px 18px rgba(33, 48, 92, 0.035);
    }
    [data-testid="stChatInput"] {
        border-color: #dce1ed;
        border-radius: 14px;
        background: #ffffff;
        box-shadow: 0 6px 20px rgba(33, 48, 92, 0.05);
    }
    [data-testid="stBaseButton-primary"] {
        min-height: 42px;
        border: 0;
        border-radius: 10px;
        background: linear-gradient(100deg, #5869e8, #6d79ed);
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(86,104,232,0.2);
    }
    [data-testid="stBaseButton-primary"]:hover {
        border: 0;
        background: #495bd6;
        color: #ffffff;
    }
    @media (max-width: 700px) {
        .block-container { padding: 2rem 1rem 3rem; }
        .welcome-card { padding: 24px 18px; }
        .welcome-steps { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Providers: user picks one and enters their own API key ----------
PROVIDERS = {
    "Groq (fast, free)": {"model": "openai/gpt-oss-20b", "env": "GROQ_API_KEY"},
    "Google Gemini": {"model": "gemini-3.8-flash", "env": "GOOGLE_API_KEY"},
    "OpenAI": {"model": "gpt-4o-mini", "env": "OPENAI_API_KEY"},
    "Anthropic (Claude)": {"model": "claude-sonnet-4-5", "env": "ANTHROPIC_API_KEY"},
}

def get_llm(provider, model, key):
    if provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, api_key=key, temperature=0)
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=key, temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, api_key=key, temperature=0)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">📚</div>
            <div>
                <div class="brand-name">DocuMind</div>
                <div class="brand-subtitle">Document intelligence</div>
            </div>
        </div>
        <div class="section-label">01 · AI assistant</div>
        """,
        unsafe_allow_html=True,
    )
    provider = st.selectbox("AI provider", list(PROVIDERS))
    model = st.text_input("Model name", value=PROVIDERS[provider]["model"], key=f"model_{provider}")
    key = st.text_input(f"{provider} API key", type="password",
                        value=os.environ.get(PROVIDERS[provider]["env"], ""),
                        help="Read from your environment or entered for this session; never saved in the code.")
    st.markdown('<div class="section-label">02 · Knowledge source</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "md"])
    build = st.button("Create knowledge base", type="primary", disabled=upload is None, use_container_width=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

@st.cache_resource
def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_documents(uploaded):
    """Return a list of LangChain Documents. 'page' metadata = page no. (PDF) or section no."""
    ext = uploaded.name.lower().rsplit(".", 1)[-1]
    data = uploaded.getvalue()
    if ext == "pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(data); path = f.name
        try:
            return PyMuPDFLoader(path).load()
        finally:
            os.unlink(path)
    if ext == "docx":
        import docx
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
            f.write(data); path = f.name
        try:
            paras = [p.text for p in docx.Document(path).paragraphs if p.text.strip()]
        finally:
            os.unlink(path)
        text = "\n".join(paras)
    else:
        text = data.decode("utf-8", errors="ignore")
    # Split plain text into ~3000-char "sections" so we can cite them
    parts = [text[i:i + 3000] for i in range(0, len(text), 3000)]
    return [Document(page_content=p, metadata={"page": i}) for i, p in enumerate(parts) if p.strip()]

if "store" not in st.session_state: st.session_state.store = None
if "history" not in st.session_state: st.session_state.history = []

# ---------- Workspace header ----------
st.markdown('<div class="eyebrow">DOCUMENT INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
st.title("Ask your documents.")
st.markdown(
    '<div class="hero-copy">Turn your files into a focused knowledge base. Ask questions in natural language and get answers grounded in your source material.</div>',
    unsafe_allow_html=True,
)

if st.session_state.store:
    metric_doc, metric_chunks, metric_status = st.columns([2, 1, 1])
    metric_doc.metric("Active document", st.session_state.get("docname", "Uploaded file"))
    metric_chunks.metric("Indexed sections", st.session_state.get("chunk_count", 0))
    metric_status.metric("Workspace", "Ready")
    st.markdown("### Conversation")
else:
    st.markdown(
        """
        <div class="welcome-card">
            <div class="welcome-icon">✨</div>
            <div class="welcome-title">Your next answer starts with a document</div>
            <div class="welcome-text">
                Add an AI provider and API key, then upload a PDF, DOCX, TXT, or Markdown file.
                We’ll build a searchable knowledge base so you can ask questions with answers tied to your content.
            </div>
            <div class="welcome-steps">
                <div class="welcome-step"><strong>1. Choose a model</strong>Select a provider and add its API key in the sidebar.</div>
                <div class="welcome-step"><strong>2. Add a source</strong>Upload the document you want to explore.</div>
                <div class="welcome-step"><strong>3. Start asking</strong>Create the knowledge base, then ask away.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Build knowledge base ----------
if build and upload:
    with st.spinner("Reading document and creating embeddings..."):
        try:
            docs = load_documents(upload)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
            if not chunks:
                st.error("No text found. If it is a scanned PDF, try a text-based one.")
            else:
                st.session_state.store = FAISS.from_documents(chunks, embeddings())
                st.session_state.docname = upload.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.history = []
                st.success(f"Loaded {len(docs)} pages/sections and indexed {len(chunks)} chunks.")
                st.rerun()
        except Exception as e:
            st.error(f"Could not read the document: {e}")

# ---------- Chat ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer only from the supplied document context. "
               "If the answer is not present, say you could not find it in the document. "
               "Use simple language and cite source page numbers."),
    ("human", "Question: {question}\n\nDocument context:\n{context}"),
])

if st.session_state.store:
    q = st.chat_input("Ask a question about the document...")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        if not key:
            st.warning("Enter your API key in the sidebar first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        docs = st.session_state.store.similarity_search(q, k=3)
                        context = "\n\n".join(
                            f"[Source page {d.metadata.get('page', 0) + 1}]\n{d.page_content}" for d in docs)
                        llm = get_llm(provider, model, key)
                        answer = st.write_stream(
                            (PROMPT | llm | StrOutputParser()).stream({"question": q, "context": context}))
                        with st.expander("Sources"):
                            for d in docs:
                                st.write(f"Page {d.metadata.get('page', 0) + 1}: {d.page_content[:700]}")
                        st.session_state.history.append({"role": "assistant", "content": answer})
                    except ImportError as e:
                        st.error(f"Missing package for {provider}: {e}. Run: pip install -r requirements.txt")
                    except Exception as e:
                        st.error(f"AI call failed (check API key / model name): {e}")
else:
    st.caption("Supported formats: PDF · DOCX · TXT · Markdown")
