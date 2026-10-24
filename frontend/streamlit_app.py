"""
Multimodal RAG Knowledge Assistant — Streamlit Frontend
Professional UI redesign: cohesive design system, proper hierarchy,
accessible contrast, responsive layout, real data only.

Pages:
1. Chat with Documents
2. Document Library
3. Retrieval Explorer
4. Evaluation Lab
5. Settings / Diagnostics
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Multimodal RAG Knowledge Assistant — Portfolio Project"},
)

# ── Global CSS design system ─────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Design tokens ── */
:root {
    --blue-600: #2563eb;
    --blue-700: #1d4ed8;
    --blue-50:  #eff6ff;
    --blue-100: #dbeafe;
    --slate-50:  #f8fafc;
    --slate-100: #f1f5f9;
    --slate-200: #e2e8f0;
    --slate-400: #94a3b8;
    --slate-500: #64748b;
    --slate-700: #334155;
    --slate-900: #0f172a;
    --green-50:  #f0fdf4;
    --green-600: #16a34a;
    --amber-50:  #fffbeb;
    --amber-600: #d97706;
    --red-50:    #fef2f2;
    --red-600:   #dc2626;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.04);
    --shadow-md: 0 4px 12px rgba(0,0,0,.08), 0 2px 4px rgba(0,0,0,.04);
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--slate-900) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label {
    padding: 9px 12px !important;
    border-radius: var(--radius-sm) !important;
    transition: background 0.15s !important;
    font-size: 0.9rem !important;
    cursor: pointer !important;
    display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label {
    background: rgba(37,99,235,0.35) !important;
    color: #bfdbfe !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
    margin: 12px 0 !important;
}

/* ── Main area ── */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Page headings ── */
h1 { font-size: 1.6rem !important; font-weight: 700 !important;
     color: var(--slate-900) !important; letter-spacing: -0.3px !important;
     margin-bottom: 4px !important; }
h2 { font-size: 1.15rem !important; font-weight: 600 !important;
     color: var(--slate-700) !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important;
     color: var(--slate-700) !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: white !important;
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px 20px !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="metric-container"] label {
    font-size: 0.78rem !important; font-weight: 600 !important;
    color: var(--slate-500) !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700 !important;
    color: var(--blue-600) !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: var(--blue-600) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: background 0.15s, transform 0.1s !important;
    box-shadow: var(--shadow-sm) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--blue-700) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--slate-700) !important;
    background: white !important;
    font-size: 0.88rem !important;
    transition: all 0.15s !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: var(--blue-600) !important;
    color: var(--blue-600) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    background: white !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 8px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--slate-700) !important;
    padding: 12px 16px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid var(--slate-200) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: var(--slate-500) !important;
    padding: 10px 18px !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue-600) !important;
    border-bottom-color: var(--blue-600) !important;
}

/* ── Input fields ── */
.stTextInput input, .stTextArea textarea {
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--blue-600) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Select / multiselect ── */
.stSelectbox > div, .stMultiSelect > div {
    border-radius: var(--radius-sm) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left-width: 4px !important;
    font-size: 0.88rem !important;
}

/* ── Tables ── */
.stDataFrame, [data-testid="stTable"] {
    border: 1px solid var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--slate-200) !important;
    padding: 12px 16px !important;
    margin-bottom: 10px !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Divider ── */
hr { border-color: var(--slate-200) !important; margin: 20px 0 !important; }

/* ── Caption / small text ── */
.stCaption { color: var(--slate-500) !important; font-size: 0.78rem !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: var(--blue-600) !important;
    border-radius: 4px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--slate-200) !important;
    border-radius: var(--radius-md) !important;
    background: var(--slate-50) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--blue-600) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] {
    background: var(--blue-600) !important;
}

/* ── Badge helper ── */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.73rem; font-weight: 600;
    letter-spacing: 0.2px;
}
.badge-blue  { background: var(--blue-50);  color: var(--blue-700); }
.badge-green { background: var(--green-50); color: var(--green-600); }
.badge-amber { background: var(--amber-50); color: var(--amber-600); }
.badge-red   { background: var(--red-50);   color: var(--red-600); }
.badge-slate { background: var(--slate-100); color: var(--slate-700); }

/* ── Card ── */
.card {
    background: white; border: 1px solid var(--slate-200);
    border-radius: var(--radius-lg); padding: 20px 24px;
    box-shadow: var(--shadow-sm); margin-bottom: 12px;
}

/* ── Score bar ── */
.score-row { display: flex; align-items: center; gap: 10px; margin: 4px 0; }
.score-label { font-size: 0.78rem; color: var(--slate-500);
               font-weight: 600; width: 120px; flex-shrink: 0; }
.score-bar-wrap { flex: 1; background: var(--slate-100);
                  border-radius: 4px; height: 6px; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 4px;
                  background: var(--blue-600); transition: width 0.3s; }
.score-val { font-size: 0.78rem; font-weight: 700; color: var(--slate-700);
             width: 50px; text-align: right; }

/* ── Latency pill ── */
.latency-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }
.latency-pill { background: var(--slate-100); border-radius: 20px;
                padding: 3px 10px; font-size: 0.73rem; color: var(--slate-600);
                font-weight: 500; }
.latency-pill span { font-weight: 700; color: var(--slate-900); }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0 0 12px; margin-bottom: 16px;
    border-bottom: 2px solid var(--slate-100);
}
.section-header h2 { margin: 0 !important; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 48px 24px;
    color: var(--slate-400);
}
.empty-state .empty-icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 { color: var(--slate-500) !important; font-size: 1rem !important; }
.empty-state p { font-size: 0.88rem; margin-top: 6px; }

/* ── Doc row ── */
.doc-row {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 18px; background: white;
    border: 1px solid var(--slate-200); border-radius: var(--radius-md);
    margin-bottom: 8px; box-shadow: var(--shadow-sm);
    transition: box-shadow 0.15s;
}
.doc-row:hover { box-shadow: var(--shadow-md); }
.doc-type-pill {
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800; flex-shrink: 0;
}

/* ── Status indicator ── */
.status-dot { width: 8px; height: 8px; border-radius: 50%;
              display: inline-block; margin-right: 6px; }
.status-ok { background: #22c55e; }
.status-warn { background: #f59e0b; }
.status-err { background: #ef4444; }

/* ── Sidebar brand ── */
.sidebar-brand {
    font-size: 1rem; font-weight: 700; color: white !important;
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0 16px;
}
.sidebar-brand-sub {
    font-size: 0.72rem; color: #94a3b8 !important;
    margin-top: -10px; padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Backend initialization ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Initializing RAG system…")
def init_backend():
    """Initialize all backend components once (cached across reruns)."""
    from app.config.settings import settings
    from app.config.logging_config import setup_logging
    setup_logging("WARNING")
    from app.api.dependencies import _init_components, get_rag_pipeline, get_document_service
    _init_components()
    return get_rag_pipeline(), get_document_service(), settings


def get_backend():
    return init_backend()


# ── Session state defaults ───────────────────────────────────────────────────

for key, default in [
    ("conversation_history", []),
    ("chat_messages", []),
    ("last_retrieval_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🔍 RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-brand-sub">Knowledge Intelligence System</div>', unsafe_allow_html=True)
    st.markdown("---")

    PAGES = [
        "💬 Chat",
        "📚 Document Library",
        "🔎 Retrieval Explorer",
        "📊 Evaluation Lab",
        "⚙️ Settings",
    ]
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    # Quick status in sidebar
    st.markdown("---")
    try:
        from app.api.dependencies import _vector_store, _document_store
        if _vector_store is not None:
            n_docs = _document_store.count() if _document_store else 0
            n_chunks = _vector_store.count()
            st.markdown(
                f'<div style="font-size:0.75rem;color:#94a3b8;padding:4px 0;">'
                f'<span style="color:#4ade80">●</span> {n_docs} docs · {n_chunks} chunks indexed</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.75rem;color:#94a3b8;padding:4px 0;">'
                '<span style="color:#f59e0b">●</span> Index not initialized</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: CHAT
# ════════════════════════════════════════════════════════════════════════════

if page == PAGES[0]:
    # ── Header ──
    col_title, col_actions = st.columns([6, 2])
    with col_title:
        st.title("💬 Chat with Documents")
        st.caption("Ask questions about your indexed documents. Answers are grounded with source citations.")

    with col_actions:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            st.rerun()

    st.markdown("---")

    # ── Two-column layout: controls left, chat right ──
    ctrl_col, chat_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### ⚙️ Options")

        # Upload
        st.markdown("**Upload documents**")
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "docx", "txt", "md", "csv", "pptx", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="chat_uploader",
            label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("📥 Ingest Files", type="primary", use_container_width=True):
                try:
                    _, svc, _ = get_backend()
                    all_ok = True
                    for uf in uploaded_files:
                        with st.spinner(f"Processing {uf.name}…"):
                            content = uf.read()
                            saved = svc.save_upload(uf.name, content)
                            result = svc.ingest_file(saved)
                            if result["status"] == "success":
                                st.success(
                                    f"✓ **{result['file_name']}** — "
                                    f"{result['chunks_added']} chunks"
                                )
                            elif result["status"] == "duplicate":
                                st.info(f"↩ {result['file_name']} already indexed")
                            else:
                                st.error(f"✗ {result['file_name']}: {result.get('error_message') or 'unknown error'}")
                                all_ok = False
                    if all_ok:
                        st.rerun()
                except Exception as e:
                    st.error(f"Ingestion error: {e}")

        st.markdown("---")
        st.markdown("**Retrieval**")
        retrieval_method = st.selectbox(
            "Method",
            ["hybrid", "dense", "sparse"],
            help="**hybrid** = dense + BM25 via Reciprocal Rank Fusion\n**dense** = semantic similarity only\n**sparse** = BM25 keyword matching",
        )
        query_strategy = st.selectbox(
            "Query strategy",
            ["none", "rewrite", "expand", "multi_query", "decompose", "hyde"],
            help="Transforms the query before retrieval to improve results",
        )
        top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5,
                          help="How many chunks to retrieve and pass to the LLM")

        # Strategy explanation
        strategy_desc = {
            "none": "Query used as-is",
            "rewrite": "Clarify and fix query phrasing",
            "expand": "Add synonyms and related terms",
            "multi_query": "Generate 3 query variants, merge results",
            "decompose": "Break complex question into sub-questions",
            "hyde": "Generate hypothetical answer for embedding",
        }
        if query_strategy != "none":
            st.markdown(
                f'<div style="background:#eff6ff;border-radius:6px;padding:8px 12px;'
                f'font-size:0.78rem;color:#1d4ed8;margin-top:4px;">ℹ️ {strategy_desc[query_strategy]}</div>',
                unsafe_allow_html=True,
            )

    with chat_col:
        # ── Empty state ──
        if not st.session_state.chat_messages:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">💬</div>'
                '<h3>Start a conversation</h3>'
                '<p>Upload documents using the panel on the left, then ask questions here.<br>'
                'Answers include numbered citations linked to source documents.</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Chat history ──
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    # Latency row
                    if msg.get("latency"):
                        lat = msg["latency"]
                        st.markdown(
                            f'<div class="latency-row">'
                            f'<div class="latency-pill">⚡ Retrieval <span>{lat.get("retrieval", 0):.2f}s</span></div>'
                            f'<div class="latency-pill">✍️ Generation <span>{lat.get("generation", 0):.2f}s</span></div>'
                            f'<div class="latency-pill">🕐 Total <span>{lat.get("total", 0):.2f}s</span></div>'
                            f'<div class="latency-pill">📋 Method <span>{lat.get("method", "?")}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    # Sources
                    if msg.get("sources"):
                        with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                            for s in msg["sources"]:
                                st.markdown(
                                    f'<div style="display:flex;align-items:center;gap:10px;'
                                    f'padding:6px 0;border-bottom:1px solid #f1f5f9;">'
                                    f'<span class="badge badge-blue">[{s["citation_num"]}]</span>'
                                    f'<span style="font-size:0.85rem;font-weight:600;">{s["source"]}</span>'
                                    f'<span style="font-size:0.78rem;color:#64748b;">page {s["page_number"]}</span>'
                                    f'<span style="margin-left:auto;font-size:0.75rem;color:#94a3b8;">'
                                    f'score {s["score"]:.3f}</span></div>',
                                    unsafe_allow_html=True,
                                )
                    # Citation warnings
                    if msg.get("citation_warnings"):
                        with st.expander("⚠️ Citation warnings"):
                            for w in msg["citation_warnings"]:
                                st.warning(w)
                    # Insufficient evidence
                    if not msg.get("evidence_sufficient", True):
                        st.info("ℹ️ The indexed documents do not contain sufficient evidence for this question.")
                    # Retrieved chunks
                    if msg.get("retrieval_results"):
                        with st.expander(f"🔎 Retrieved chunks ({len(msg['retrieval_results'])})"):
                            for i, r in enumerate(msg["retrieval_results"][:5]):
                                meta = r.get("metadata", {})
                                src = meta.get("source", r.get("source", "?"))
                                pg  = meta.get("page_number", r.get("page_number", "?"))
                                sc  = r.get("score", 0.0)
                                st.markdown(
                                    f'<div style="display:flex;justify-content:space-between;'
                                    f'margin-bottom:4px;">'
                                    f'<span style="font-size:0.82rem;font-weight:600;">#{i+1} {src} · p.{pg}</span>'
                                    f'<span class="badge badge-slate">score {sc:.3f}</span></div>',
                                    unsafe_allow_html=True,
                                )
                                st.code(r["text"][:400], language=None)
                                if i < 4:
                                    st.markdown('<hr style="margin:8px 0;">', unsafe_allow_html=True)

        # ── Chat input ──
        if prompt := st.chat_input("Ask a question about your documents…"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving evidence and generating answer…"):
                    try:
                        pipeline, _, _ = get_backend()
                        t_start = time.time()
                        response = pipeline.query(
                            question=prompt,
                            retrieval_method=retrieval_method,
                            query_strategy=query_strategy,
                            conversation_history=st.session_state.conversation_history,
                            top_k=top_k,
                        )
                        elapsed = time.time() - t_start

                        st.markdown(response.answer)

                        lat = {
                            "retrieval": response.retrieval_latency_s,
                            "generation": response.generation_latency_s,
                            "total": elapsed,
                            "method": f"{retrieval_method}/{query_strategy}",
                        }
                        st.markdown(
                            f'<div class="latency-row">'
                            f'<div class="latency-pill">⚡ Retrieval <span>{lat["retrieval"]:.2f}s</span></div>'
                            f'<div class="latency-pill">✍️ Generation <span>{lat["generation"]:.2f}s</span></div>'
                            f'<div class="latency-pill">🕐 Total <span>{lat["total"]:.2f}s</span></div>'
                            f'<div class="latency-pill">📋 Method <span>{lat["method"]}</span></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if response.sources:
                            with st.expander(f"📎 {len(response.sources)} source(s)"):
                                for s in response.sources:
                                    st.markdown(
                                        f'<div style="display:flex;align-items:center;gap:10px;'
                                        f'padding:6px 0;border-bottom:1px solid #f1f5f9;">'
                                        f'<span class="badge badge-blue">[{s["citation_num"]}]</span>'
                                        f'<span style="font-size:0.85rem;font-weight:600;">{s["source"]}</span>'
                                        f'<span style="font-size:0.78rem;color:#64748b;">page {s["page_number"]}</span>'
                                        f'<span style="margin-left:auto;font-size:0.75rem;color:#94a3b8;">'
                                        f'score {s["score"]:.3f}</span></div>',
                                        unsafe_allow_html=True,
                                    )

                        if response.citation_warnings:
                            with st.expander("⚠️ Citation warnings"):
                                for w in response.citation_warnings:
                                    st.warning(w)

                        if not response.evidence_sufficient:
                            st.info("ℹ️ The indexed documents do not contain sufficient evidence for this question.")

                        if response.retrieval_results:
                            with st.expander(f"🔎 Retrieved chunks ({len(response.retrieval_results)})"):
                                for i, r in enumerate(response.retrieval_results[:5]):
                                    meta = r.get("metadata", {})
                                    src = meta.get("source", r.get("source", "?"))
                                    pg  = meta.get("page_number", r.get("page_number", "?"))
                                    sc  = r.get("score", 0.0)
                                    st.markdown(
                                        f'<div style="display:flex;justify-content:space-between;'
                                        f'margin-bottom:4px;">'
                                        f'<span style="font-size:0.82rem;font-weight:600;">#{i+1} {src} · p.{pg}</span>'
                                        f'<span class="badge badge-slate">score {sc:.3f}</span></div>',
                                        unsafe_allow_html=True,
                                    )
                                    st.code(r["text"][:400], language=None)
                                    if i < 4:
                                        st.markdown('<hr style="margin:8px 0;">', unsafe_allow_html=True)

                        # Bounded conversation history
                        st.session_state.conversation_history.extend([
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response.answer},
                        ])
                        st.session_state.conversation_history = \
                            st.session_state.conversation_history[-20:]

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "citation_warnings": response.citation_warnings,
                            "evidence_sufficient": response.evidence_sufficient,
                            "retrieval_results": response.retrieval_results,
                            "latency": lat,
                        })

                    except Exception as e:
                        st.error(f"**Error:** {e}")
                        st.info("Check Settings to verify your API keys and index status.")
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": f"❌ Error: {e}",
                        })


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: DOCUMENT LIBRARY
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    st.title("📚 Document Library")
    st.caption("Manage documents indexed in the vector store.")
    st.markdown("---")

    try:
        _, svc, _ = get_backend()
        docs = svc.list_documents()

        # ── Metrics row ──
        col1, col2, col3, col4 = st.columns(4)
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        total_tables = sum(d.get("table_count", 0) for d in docs)
        total_vis    = sum(d.get("visual_count", 0) for d in docs)
        col1.metric("Documents", len(docs))
        col2.metric("Chunks", total_chunks)
        col3.metric("Tables", total_tables)
        col4.metric("Visuals", total_vis)
        st.markdown("---")

        if not docs:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">📂</div>'
                '<h3>No documents indexed</h3>'
                '<p>Upload files below to get started. Supported formats: PDF, DOCX, TXT, MD, CSV, PPTX, PNG, JPG.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            # ── Document list ──
            st.markdown("#### Indexed Documents")
            TYPE_COLORS = {
                "pdf":  ("#fee2e2", "#991b1b"),
                "docx": ("#dbeafe", "#1e40af"),
                "txt":  ("#f1f5f9", "#334155"),
                "md":   ("#e0f2fe", "#0369a1"),
                "csv":  ("#dcfce7", "#166534"),
                "pptx": ("#fef3c7", "#92400e"),
                "png":  ("#f5f3ff", "#5b21b6"),
                "jpg":  ("#f5f3ff", "#5b21b6"),
                "jpeg": ("#f5f3ff", "#5b21b6"),
            }

            for doc in docs:
                ft   = doc.get("file_type", "?").lower()
                bg, fg = TYPE_COLORS.get(ft, ("#f1f5f9", "#334155"))
                doc_id = doc.get("doc_id", "")

                col_info, col_stats, col_del = st.columns([5, 3, 1])
                with col_info:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:12px;padding:4px 0;">'
                        f'<div class="doc-type-pill" style="background:{bg};color:{fg};">'
                        f'{ft.upper()}</div>'
                        f'<div><div style="font-weight:600;font-size:0.9rem;">'
                        f'{doc.get("file_name", "Unknown")}</div>'
                        f'<div style="font-size:0.75rem;color:#94a3b8;font-family:monospace;">'
                        f'{doc_id}</div></div></div>',
                        unsafe_allow_html=True,
                    )
                with col_stats:
                    chunks = doc.get("chunk_count", 0)
                    tables = doc.get("table_count", 0)
                    visuals = doc.get("visual_count", 0)
                    st.markdown(
                        f'<div style="display:flex;gap:8px;align-items:center;padding:8px 0;">'
                        f'<span class="badge badge-slate">📄 {chunks} chunks</span>'
                        f'<span class="badge badge-slate">📊 {tables} tables</span>'
                        f'<span class="badge badge-slate">🖼 {visuals} visuals</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete this document"):
                        confirm_key = f"confirm_del_{doc_id}"
                        if st.session_state.get(confirm_key):
                            deleted = svc.delete_document(doc_id)
                            if deleted:
                                st.session_state.pop(confirm_key, None)
                                st.success("Document deleted.")
                                st.rerun()
                            else:
                                st.error("Deletion failed.")
                        else:
                            st.session_state[confirm_key] = True

                if st.session_state.get(f"confirm_del_{doc_id}"):
                    st.warning(
                        f"⚠️ Delete **{doc.get('file_name')}**? This cannot be undone. "
                        "Click 🗑️ again to confirm."
                    )
                st.markdown('<hr style="margin:6px 0;">', unsafe_allow_html=True)

        # ── Upload section ──
        st.markdown("---")
        st.markdown("#### Upload New Document")
        uf = st.file_uploader(
            "Select a file to upload and ingest",
            type=["pdf", "docx", "txt", "md", "csv", "pptx", "png", "jpg", "jpeg"],
            key="lib_uploader",
        )
        if uf:
            col_btn, col_info2 = st.columns([2, 5])
            with col_btn:
                if st.button("📥 Upload & Ingest", type="primary", use_container_width=True):
                    with st.spinner(f"Processing {uf.name}…"):
                        try:
                            content = uf.read()
                            saved = svc.save_upload(uf.name, content)
                            result = svc.ingest_file(saved)
                            if result["status"] == "success":
                                st.success(
                                    f"✓ **{result['file_name']}** ingested — "
                                    f"{result['chunks_added']} chunks, "
                                    f"{result['tables_indexed']} tables, "
                                    f"{result['visuals_indexed']} visuals"
                                )
                                st.rerun()
                            elif result["status"] == "duplicate":
                                st.info("This document is already indexed.")
                            else:
                                st.error(f"Ingestion failed: {result.get('error_message') or 'unknown error'}")
                        except ValueError as e:
                            st.error(str(e))
            with col_info2:
                st.caption(f"Selected: **{uf.name}** ({uf.size/1024:.1f} KB)")

    except Exception as e:
        st.error(f"Error loading document library: {e}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: RETRIEVAL EXPLORER
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    st.title("🔎 Retrieval Explorer")
    st.caption(
        "Compare retrieval strategies side-by-side for the same query. "
        "Inspect scores, chunk content, and ranking metadata."
    )
    st.markdown("---")

    # ── Controls ──
    query = st.text_input(
        "Query",
        placeholder="e.g. What is the RRF k constant and why is it used?",
        label_visibility="collapsed",
    )

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col1:
        methods = st.multiselect(
            "Retrieval methods to compare",
            ["dense", "sparse", "hybrid"],
            default=["dense", "sparse", "hybrid"],
        )
    with col2:
        exp_top_k = st.slider("Top-K", 1, 20, 10)
    with col3:
        show_text = st.checkbox("Show chunk text", value=True)
    with col4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Comparison", type="primary", use_container_width=True,
                            disabled=not (query and methods))

    if not query:
        st.markdown(
            '<div class="empty-state" style="padding:32px 0;">'
            '<div class="empty-icon">🔎</div>'
            '<h3>Enter a query above</h3>'
            '<p>Type a question and select retrieval methods to compare their results.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    elif not methods:
        st.warning("Select at least one retrieval method.")
    elif run_btn:
        try:
            pipeline, _, _ = get_backend()
            all_results: dict = {}
            METHOD_ICONS = {"dense": "🔵", "sparse": "🟡", "hybrid": "🟢"}

            progress = st.progress(0, text="Running retrieval…")
            for i, method in enumerate(methods):
                t = time.time()
                results = pipeline._retrieve(query, method=method, top_k=exp_top_k)
                elapsed = time.time() - t
                all_results[method] = {"results": results, "latency": elapsed}
                progress.progress((i + 1) / len(methods), text=f"✓ {method} done")
            progress.empty()

            # ── Results comparison ──
            tabs = st.tabs([
                f"{METHOD_ICONS.get(m,'•')} {m.upper()}  ·  {len(all_results[m]['results'])} results  ·  {all_results[m]['latency']*1000:.0f}ms"
                for m in methods
            ])

            for tab, method in zip(tabs, methods):
                with tab:
                    info = all_results[method]
                    results_list = info["results"]

                    if not results_list:
                        st.markdown(
                            '<div class="empty-state" style="padding:24px 0;">'
                            '<div class="empty-icon">🔍</div>'
                            '<h3>No results returned</h3>'
                            '<p>This method found no matches. Try a different query or ingest documents first.</p>'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        continue

                    # Score distribution summary
                    scores = [r.get("score", 0.0) for r in results_list]
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        max_score = max(scores)
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Results", len(results_list))
                        col_b.metric("Max Score", f"{max_score:.4f}")
                        col_c.metric("Avg Score", f"{avg_score:.4f}")

                    st.markdown("---")

                    for i, r in enumerate(results_list):
                        meta       = r.get("metadata", {})
                        src        = meta.get("source", r.get("source", "?"))
                        pg         = meta.get("page_number", r.get("page_number", "?"))
                        score      = r.get("score", 0.0)
                        rrf_score  = r.get("rrf_score")
                        rerank_sc  = r.get("reranker_score")
                        retrieval_m = r.get("retrieval_method", method)
                        chunk_id   = r.get("chunk_id", "?")

                        label = f"#{i+1}  {src}  ·  page {pg}"
                        with st.expander(label, expanded=(i == 0)):
                            left, right = st.columns(2)
                            with left:
                                st.markdown(
                                    f'<div style="font-size:0.82rem;line-height:2;">'
                                    f'<b>Chunk ID:</b> <code style="font-size:0.75rem;">{chunk_id}</code><br>'
                                    f'<b>Source:</b> {src}<br>'
                                    f'<b>Page:</b> {pg}<br>'
                                    f'<b>Method:</b> <code>{retrieval_m}</code>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                            with right:
                                bar_pct = min(int(score * 100), 100)
                                st.markdown(
                                    f'<div class="score-row"><div class="score-label">Score</div>'
                                    f'<div class="score-bar-wrap"><div class="score-bar-fill" style="width:{bar_pct}%"></div></div>'
                                    f'<div class="score-val">{score:.4f}</div></div>',
                                    unsafe_allow_html=True,
                                )
                                if rrf_score is not None:
                                    rrf_pct = min(int(rrf_score * 10000), 100)
                                    st.markdown(
                                        f'<div class="score-row"><div class="score-label">RRF Score</div>'
                                        f'<div class="score-bar-wrap"><div class="score-bar-fill" '
                                        f'style="width:{rrf_pct}%;background:#7c3aed;"></div></div>'
                                        f'<div class="score-val">{rrf_score:.5f}</div></div>',
                                        unsafe_allow_html=True,
                                    )
                                if rerank_sc is not None:
                                    st.markdown(
                                        f'<div class="score-row"><div class="score-label">Reranker</div>'
                                        f'<div class="score-bar-wrap"><div class="score-bar-fill" '
                                        f'style="background:#16a34a;width:{min(int((rerank_sc+10)*5),100)}%;"></div></div>'
                                        f'<div class="score-val">{rerank_sc:.3f}</div></div>',
                                        unsafe_allow_html=True,
                                    )

                            if show_text and r.get("text"):
                                st.markdown("**Chunk text:**")
                                st.code(r["text"][:600], language=None)

        except Exception as e:
            st.error(f"Retrieval error: {e}")
            st.info("Make sure documents are ingested before running retrieval.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: EVALUATION LAB
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    st.title("📊 Evaluation Lab")
    st.caption(
        "Run retrieval benchmarks against the evaluation dataset. "
        "All results are computed from actual retrieval — no fabricated numbers."
    )
    st.markdown("---")

    eval_path = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"

    if not eval_path.exists():
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">📋</div>'
            '<h3>No evaluation dataset found</h3>'
            '<p>Run the following command to generate the demo corpus and evaluation dataset:</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.code("python scripts/create_demo_corpus.py", language="bash")
    else:
        from app.evaluation.dataset import load_eval_dataset
        cases = load_eval_dataset(eval_path)

        # ── Dataset overview ──
        easy   = sum(1 for c in cases if c.difficulty == "easy")
        medium = sum(1 for c in cases if c.difficulty == "medium")
        hard   = sum(1 for c in cases if c.difficulty == "hard")
        text_q = sum(1 for c in cases if c.answer_type == "text")
        table_q= sum(1 for c in cases if c.answer_type == "table")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Cases", len(cases))
        col2.metric("Easy", easy)
        col3.metric("Medium", medium)
        col4.metric("Hard", hard)
        col5.metric("Table Q's", table_q)

        with st.expander("📋 View evaluation cases"):
            for c in cases:
                difficulty_badge = {
                    "easy": "badge-green", "medium": "badge-amber", "hard": "badge-red"
                }.get(c.difficulty, "badge-slate")
                type_badge = "badge-blue" if c.answer_type == "table" else "badge-slate"
                st.markdown(
                    f'<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 0;'
                    f'border-bottom:1px solid #f1f5f9;">'
                    f'<span class="badge {difficulty_badge}">{c.difficulty}</span>'
                    f'<span class="badge {type_badge}">{c.answer_type}</span>'
                    f'<span style="font-size:0.85rem;flex:1;">'
                    f'<b>{c.case_id}</b>: {c.question}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Benchmark controls ──
        st.markdown("#### Run Retrieval Benchmark")
        col_a, col_b = st.columns([3, 2])
        with col_a:
            configs_to_run = st.multiselect(
                "Select configurations",
                ["dense", "sparse_bm25", "hybrid_rrf"],
                default=["dense", "hybrid_rrf"],
                help="Each config will be evaluated on all cases",
            )
        with col_b:
            k = st.slider("Metric cutoff k", 1, 10, 5,
                          help="Metrics are computed at rank k (e.g., Hit@5)")

        if st.button("▶ Run Benchmark", type="primary",
                     disabled=not configs_to_run) and configs_to_run:
            try:
                pipeline, _, _ = get_backend()
                from app.evaluation.benchmark import run_retrieval_benchmark

                config_fns = {
                    "dense":       lambda q: pipeline.dense.retrieve(q, top_k=k),
                    "sparse_bm25": lambda q: pipeline.sparse.retrieve(q, top_k=k),
                    "hybrid_rrf":  lambda q: pipeline.hybrid.retrieve(q, top_k=k),
                }

                bench_results = []
                progress = st.progress(0, text="Starting benchmark…")
                for i, name in enumerate(configs_to_run):
                    progress.progress(i / len(configs_to_run), text=f"Benchmarking {name}…")
                    r = run_retrieval_benchmark(cases, config_fns[name], name, k=k)
                    bench_results.append(r)
                progress.progress(1.0, text="Done!")
                progress.empty()

                # ── Results table ──
                st.markdown("#### Results")
                rows = []
                for r in bench_results:
                    ag = r.aggregated
                    rows.append({
                        "Configuration": r.config_name,
                        "MRR": round(ag.get("mrr", 0), 3),
                        f"Hit@{k}": round(ag.get(f"hit@{k}", 0), 3),
                        f"Recall@{k}": round(ag.get(f"recall@{k}", 0), 3),
                        f"nDCG@{k}": round(ag.get(f"ndcg@{k}", 0), 3),
                        "Cases": r.cases_run,
                        "Latency (s)": round(r.total_latency_s, 2),
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

                # ── Score bars for MRR ──
                st.markdown("**MRR comparison:**")
                for r in bench_results:
                    mrr = r.aggregated.get("mrr", 0)
                    bar_pct = int(mrr * 100)
                    st.markdown(
                        f'<div class="score-row" style="margin:6px 0;">'
                        f'<div class="score-label" style="width:140px;">{r.config_name}</div>'
                        f'<div class="score-bar-wrap"><div class="score-bar-fill" style="width:{bar_pct}%;"></div></div>'
                        f'<div class="score-val">{mrr:.3f}</div></div>',
                        unsafe_allow_html=True,
                    )

                # ── Export ──
                export_data = [
                    {
                        "config_name": r.config_name,
                        "aggregated_metrics": r.aggregated,
                        "per_case_metrics": r.per_case_metrics,
                    }
                    for r in bench_results
                ]
                col_dl, _ = st.columns([2, 5])
                with col_dl:
                    st.download_button(
                        "📥 Export JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name="benchmark_results.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                # ── Errors ──
                errors = [(r.config_name, e) for r in bench_results for e in r.errors]
                if errors:
                    with st.expander(f"⚠️ {len(errors)} error(s) during benchmark"):
                        for cfg, err in errors:
                            st.warning(f"[{cfg}] {err}")

            except Exception as e:
                st.error(f"Benchmark failed: {e}")
                st.info("Make sure documents are ingested before running evaluation.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5: SETTINGS / DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    st.title("⚙️ Settings & Diagnostics")
    st.caption("Configuration status, index health, and system capabilities. API keys are never shown.")
    st.markdown("---")

    try:
        pipeline, svc, settings = get_backend()

        # ── Capability cards ──
        st.markdown("#### System Capabilities")
        caps = settings.report_capabilities()

        cap_cols = st.columns(4)
        cap_items = list(caps.items())
        CAPABILITY_ICONS = {
            "llm_generation": "🧠",
            "vision_analysis": "👁️",
            "reranking": "🎯",
            "mmr": "📐",
            "embedding_model": "🔢",
            "llm_model": "⚡",
            "groq_model": "⚡",
            "vision_model": "🖼️",
            "reranker_model": "📊",
        }
        for i, (cap_name, cap_val) in enumerate(cap_items):
            with cap_cols[i % 4]:
                is_ok = bool(cap_val) and cap_val not in (False, "not configured", "disabled")
                status_html = (
                    '<span class="status-dot status-ok"></span>'
                    if is_ok else
                    '<span class="status-dot status-warn"></span>'
                )
                icon = CAPABILITY_ICONS.get(cap_name, "•")
                display_val = str(cap_val)
                if len(display_val) > 35:
                    display_val = display_val[:32] + "…"
                st.markdown(
                    f'<div class="card" style="padding:14px 16px;margin-bottom:10px;">'
                    f'<div style="font-size:1.2rem;margin-bottom:6px;">{icon}</div>'
                    f'<div style="font-size:0.75rem;font-weight:700;color:#64748b;'
                    f'text-transform:uppercase;letter-spacing:0.5px;">{cap_name.replace("_"," ")}</div>'
                    f'<div style="display:flex;align-items:center;gap:6px;margin-top:6px;">'
                    f'{status_html}'
                    f'<span style="font-size:0.82rem;font-weight:600;color:#334155;">{display_val}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Index status ──
        col_idx, col_cfg = st.columns(2)
        with col_idx:
            st.markdown("#### Index Status")
            try:
                from app.api.dependencies import _vector_store, _document_store
                n_chunks = _vector_store.count() if _vector_store else 0
                n_docs   = _document_store.count() if _document_store else 0
                col_a, col_b = st.columns(2)
                col_a.metric("Indexed Documents", n_docs)
                col_b.metric("Indexed Chunks", n_chunks)
                if n_docs == 0:
                    st.info("No documents indexed. Run the ingestion script or upload files in the Document Library.")
            except Exception as e:
                st.warning(f"Could not read index stats: {e}")

        with col_cfg:
            st.markdown("#### Configuration")
            cfg_data = {
                "Embedding model":  settings.embedding_model,
                "Chunk size":       settings.chunk_size,
                "Chunk overlap":    settings.chunk_overlap,
                "Top-K default":    settings.top_k,
                "Candidate-K":      settings.candidate_k,
                "Reranker":         "enabled" if settings.reranker_enabled else "disabled",
                "MMR":              "enabled" if settings.mmr_enabled else "disabled",
                "MMR lambda":       settings.mmr_lambda,
                "Max upload (MB)":  settings.max_upload_mb,
            }
            for k, v in cfg_data.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                    f'border-bottom:1px solid #f1f5f9;font-size:0.85rem;">'
                    f'<span style="color:#64748b;">{k}</span>'
                    f'<span style="font-weight:600;color:#0f172a;">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Warnings ──
        st.markdown("#### Action Required")
        warnings_list = []
        if not settings.llm_available:
            msg = (
                f"LLM answer generation is disabled ({settings.llm_provider}). "
                "The system runs in retrieval-only mode. Ensure local Ollama is running at "
                f"{settings.ollama_base_url} or configure GROQ_API_KEY."
                if settings.llm_provider == "ollama" else
                "LLM answer generation is disabled. Set GROQ_API_KEY in your .env file to enable answers."
            )
            warnings_list.append(("LLM generation not available", msg, "error"))
        if not settings.vision_available:
            msg = (
                f"Visual analysis is disabled ({settings.vision_provider}). "
                "Ensure local Ollama (Llava) is available or configure VISION_API_KEY in .env."
            )
            warnings_list.append(("Vision analysis not available", msg, "warning"))
        if not settings.reranker_enabled:
            warnings_list.append((
                "Reranking disabled",
                "Cross-encoder reranking is off. Set RERANKER_ENABLED=true in .env to enable "
                "second-stage reranking (adds ~200–400ms per query but improves precision).",
                "info",
            ))

        if not warnings_list:
            st.success("✅ All optional integrations are configured.")
        else:
            for title, msg, level in warnings_list:
                if level == "error":
                    st.error(f"**{title}** — {msg}")
                elif level == "warning":
                    st.warning(f"**{title}** — {msg}")
                else:
                    st.info(f"**{title}** — {msg}")

        # ── Quick commands ──
        st.markdown("---")
        st.markdown("#### Quick Start Commands")
        st.code(
            "# 1. Generate demo corpus\n"
            "python scripts/create_demo_corpus.py\n\n"
            "# 2. Ingest demo documents\n"
            "python scripts/ingest.py --dir data/demo_corpus/\n\n"
            "# 3. Run a test query\n"
            "python scripts/query_demo.py \"What is the RRF k constant?\"\n\n"
            "# 4. Run retrieval evaluation\n"
            "python scripts/evaluate_retrieval.py",
            language="bash",
        )

    except Exception as e:
        st.error(f"Error loading settings: {e}")
        st.info("This may indicate a dependency is not installed. Check requirements.txt.")
