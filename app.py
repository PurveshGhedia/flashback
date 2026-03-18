"""
app.py
=======
Flashback — Streamlit UI

Run with:
    streamlit run app.py

Two-state app:
  State 1 — Upload:  User uploads a lecture video → triggers ingestion pipeline
  State 2 — Chat:    Video is indexed → user asks questions → answers with
                     timestamps and keyframes are displayed

Session state keys:
  st.session_state.video_hash      : str | None  — hash of indexed video
  st.session_state.video_name      : str | None  — display name
  st.session_state.index_result    : IndexResult | None
  st.session_state.chat_history    : list[dict]  — {role, content, answer}
  st.session_state.indexing_done   : bool
"""

import streamlit as st
import logging
import time
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------
st.set_page_config(
    page_title  = "Flashback",
    page_icon   = "🎬",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ---------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* App background */
.stApp {
    background-color: #0a0a0a;
    color: #e8e8e8;
}

/* Header */
.flashback-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.flashback-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f5f5f5;
    margin: 0;
    line-height: 1;
}
.flashback-title span {
    color: #ff4d00;
}
.flashback-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* Upload zone */
.upload-zone {
    border: 1px solid #222;
    border-radius: 8px;
    padding: 2rem;
    background: #111;
    margin: 1rem 0;
}

/* Video info badge */
.video-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    font-size: 0.75rem;
    color: #888;
    font-family: 'DM Mono', monospace;
}

/* Chat messages */
.msg-user {
    background: #111;
    border: 1px solid #222;
    border-radius: 8px 8px 2px 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #ccc;
    max-width: 80%;
    margin-left: auto;
}
.msg-assistant {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #ff4d00;
    border-radius: 2px 8px 8px 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #ddd;
    line-height: 1.6;
    max-width: 90%;
}

/* Timestamp pill */
.ts-pill {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #ff4d00;
    color: #ff4d00;
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    margin-right: 0.3rem;
    letter-spacing: 0.05em;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 100px;
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ff4d00;
    display: block;
}
.stat-label {
    font-size: 0.65rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* Keyframe display */
.keyframe-container {
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}
.keyframe-timestamp {
    position: absolute;
    bottom: 0;
    left: 0;
    background: rgba(0,0,0,0.85);
    color: #ff4d00;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.4rem;
    letter-spacing: 0.05em;
}

/* Divider */
.subtle-divider {
    border: none;
    border-top: 1px solid #1a1a1a;
    margin: 1.5rem 0;
}

/* Processing steps */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.4rem 0;
    font-size: 0.8rem;
    color: #666;
    font-family: 'DM Mono', monospace;
}
.step-item.done { color: #888; }
.step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #333;
    flex-shrink: 0;
}
.step-dot.active { background: #ff4d00; }
.step-dot.done   { background: #444; }

/* Input styling override */
.stTextInput > div > div > input {
    background: #111 !important;
    border: 1px solid #222 !important;
    border-radius: 6px !important;
    color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff4d00 !important;
    box-shadow: none !important;
}

/* Button */
.stButton > button {
    background: #ff4d00 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #111 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
}

/* Scrollable chat area */
.chat-scroll {
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 0.5rem;
}

/* New chat label */
.new-label {
    font-size: 0.65rem;
    color: #333;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------
if "video_hash"     not in st.session_state: st.session_state.video_hash    = None
if "video_name"     not in st.session_state: st.session_state.video_name    = None
if "index_result"   not in st.session_state: st.session_state.index_result  = None
if "indexing_done"  not in st.session_state: st.session_state.indexing_done = False
if "chat_history"   not in st.session_state: st.session_state.chat_history  = []

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Cached pipeline functions
# (st.cache_resource — loaded once, shared across sessions)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def run_indexing(video_path: str, force: bool = False):
    """Run the full ingestion pipeline. Cached by video path."""
    from ingestion.indexer import index_video
    return index_video(video_path, force_reindex=force)


@st.cache_resource(show_spinner=False)
def get_video_hash_cached(video_path: str) -> str:
    from ingestion.indexer import get_video_hash
    return get_video_hash(video_path)


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="flashback-header">
    <div class="flashback-title">flash<span>back</span></div>
    <div class="flashback-subtitle">search inside any lecture · natural language</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)


# ---------------------------------------------------------------
# Helper: format timestamp
# ---------------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    h    = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if h > 0:
        return f"{h}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ---------------------------------------------------------------
# Helper: save uploaded file to data/videos/
# ---------------------------------------------------------------
def save_uploaded_video(uploaded_file) -> str:
    """Save Streamlit uploaded file to disk. Returns absolute path."""
    from config import VIDEOS_DIR
    save_path = VIDEOS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)


# ---------------------------------------------------------------
# Helper: render a single answer card
# ---------------------------------------------------------------
def render_answer(answer):
    """Render an Answer object as a styled chat message."""

    # Response text
    st.markdown(
        f'<div class="msg-assistant">{answer.response_text}</div>',
        unsafe_allow_html=True
    )

    # Timestamp pills
    if answer.timestamps:
        ts_html = " ".join(
            f'<span class="ts-pill">▶ {fmt_ts(t)}</span>'
            for t in answer.timestamps
        )
        st.markdown(
            f'<div style="margin: 0.5rem 0;">{ts_html}</div>',
            unsafe_allow_html=True
        )

    # Keyframes
    if answer.keyframe_paths:
        valid_paths = [p for p in answer.keyframe_paths if Path(p).exists()]
        if valid_paths:
            cols = st.columns(min(len(valid_paths), 3))
            for i, (col, path) in enumerate(zip(cols, valid_paths[:3])):
                with col:
                    try:
                        img = Image.open(path)
                        ts  = answer.timestamps[i] if i < len(answer.timestamps) else 0
                        st.image(img, caption=f"▶ {fmt_ts(ts)}", use_container_width=True)
                    except Exception:
                        pass


# ===============================================================
# STATE 1 — No video indexed yet → show upload UI
# ===============================================================
if not st.session_state.indexing_done:

    col_main, col_side = st.columns([2, 1], gap="large")

    with col_main:
        st.markdown(
            '<p style="font-size:0.75rem; color:#444; '
            'text-transform:uppercase; letter-spacing:0.12em; '
            'margin-bottom:0.8rem;">Upload Lecture Video</p>',
            unsafe_allow_html=True
        )

        uploaded = st.file_uploader(
            label       = "Drop a lecture video",
            type        = ["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility = "collapsed",
            help        = "Supported formats: MP4, MOV, AVI, MKV, WEBM",
        )

        if uploaded:
            st.markdown(
                f'<div class="video-badge">🎬 {uploaded.name} '
                f'· {uploaded.size / 1e6:.1f} MB</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Index Video →", use_container_width=False):

                # -- Save to disk --
                with st.spinner("Saving video..."):
                    video_path = save_uploaded_video(uploaded)

                # -- Run ingestion with live progress --
                st.markdown(
                    '<p style="font-size:0.75rem; color:#444; '
                    'text-transform:uppercase; letter-spacing:0.12em; '
                    'margin: 1rem 0 0.5rem 0;">Processing</p>',
                    unsafe_allow_html=True
                )

                progress_bar = st.progress(0)
                status_text  = st.empty()

                steps = [
                    (0.05, "Detecting scenes..."),
                    (0.20, "Extracting frames..."),
                    (0.40, "Filtering with SSIM..."),
                    (0.55, "Filtering with CLIP..."),
                    (0.75, "Transcribing audio with Whisper..."),
                    (0.92, "Indexing into ChromaDB..."),
                    (1.00, "Done."),
                ]

                # Animate steps while actual indexing runs in background
                # Note: Streamlit is single-threaded; progress is simulated
                # The actual indexing call blocks until complete.
                for pct, msg in steps[:-1]:
                    progress_bar.progress(pct)
                    status_text.markdown(
                        f'<span style="font-size:0.78rem; color:#666; '
                        f'font-family:\'DM Mono\',monospace;">{msg}</span>',
                        unsafe_allow_html=True
                    )
                    if pct < 0.55:
                        time.sleep(0.3)  # Fast steps
                    # Heavy steps (CLIP, Whisper) don't need fake delay

                # -- Actual indexing --
                try:
                    result = run_indexing(video_path)
                    progress_bar.progress(1.0)
                    status_text.markdown(
                        '<span style="font-size:0.78rem; color:#ff4d00; '
                        'font-family:\'DM Mono\',monospace;">✓ Ready</span>',
                        unsafe_allow_html=True
                    )

                    # Store in session state
                    st.session_state.video_hash    = result.video_hash
                    st.session_state.video_name    = uploaded.name
                    st.session_state.index_result  = result
                    st.session_state.indexing_done = True

                    time.sleep(0.5)
                    st.rerun()

                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.exception("Indexing error")

    with col_side:
        st.markdown("""
<div style="padding: 1.5rem; background: #0f0f0f; border: 1px solid #1a1a1a;
            border-radius: 8px; margin-top: 1.8rem;">
    <p style="font-family:'Syne',sans-serif; font-size:0.85rem;
              font-weight:700; color:#f5f5f5; margin:0 0 1rem 0;
              letter-spacing:-0.01em;">How it works</p>

    <div style="display:flex; flex-direction:column; gap:0.8rem;">
        <div style="display:flex; gap:0.7rem; align-items:flex-start;">
            <span style="color:#ff4d00; font-size:0.7rem; margin-top:0.1rem;">01</span>
            <p style="font-size:0.75rem; color:#555; margin:0; line-height:1.5;">
                Upload any lecture video — MP4, MOV, or MKV</p>
        </div>
        <div style="display:flex; gap:0.7rem; align-items:flex-start;">
            <span style="color:#ff4d00; font-size:0.7rem; margin-top:0.1rem;">02</span>
            <p style="font-size:0.75rem; color:#555; margin:0; line-height:1.5;">
                Flashback extracts keyframes and transcribes audio automatically</p>
        </div>
        <div style="display:flex; gap:0.7rem; align-items:flex-start;">
            <span style="color:#ff4d00; font-size:0.7rem; margin-top:0.1rem;">03</span>
            <p style="font-size:0.75rem; color:#555; margin:0; line-height:1.5;">
                Ask anything in plain English — get timestamps and explanations</p>
        </div>
    </div>
</div>
        """, unsafe_allow_html=True)


# ===============================================================
# STATE 2 — Video indexed → show chat UI
# ===============================================================
else:
    result = st.session_state.index_result

    # -- Top bar: video info + reset button --
    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.markdown(
            f'<div class="video-badge">🎬 {st.session_state.video_name} '
            f'· {result.total_keyframes} keyframes '
            f'· {result.total_chunks} transcript chunks</div>',
            unsafe_allow_html=True
        )

    with top_right:
        if st.button("← New Video", use_container_width=True):
            st.session_state.video_hash    = None
            st.session_state.video_name    = None
            st.session_state.index_result  = None
            st.session_state.indexing_done = False
            st.session_state.chat_history  = []
            st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Stats row --
    ssim_kept = (
        result.total_raw_frames - int(
            result.total_raw_frames * result.ssim_reduction_pct / 100
        )
        if result.total_raw_frames > 0 else result.total_keyframes
    )

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <span class="stat-value">{result.total_keyframes}</span>
            <div class="stat-label">Keyframes</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{result.total_chunks}</span>
            <div class="stat-label">Transcript chunks</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{result.ssim_reduction_pct:.0f}%</span>
            <div class="stat-label">SSIM reduction</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{result.clip_reduction_pct:.0f}%</span>
            <div class="stat-label">CLIP reduction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Chat history --
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:

            # User message
            st.markdown(
                f'<div class="msg-user">{entry["query"]}</div>',
                unsafe_allow_html=True
            )

            # Assistant answer
            if entry.get("answer"):
                render_answer(entry["answer"])

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Query input --
    st.markdown(
        '<p style="font-size:0.7rem; color:#333; text-transform:uppercase; '
        'letter-spacing:0.12em; margin-bottom:0.4rem;">Ask about this lecture</p>',
        unsafe_allow_html=True
    )

    input_col, btn_col = st.columns([5, 1], gap="small")

    with input_col:
        query = st.text_input(
            label            = "Query",
            placeholder      = "e.g. explain backpropagation, what is word2vec...",
            label_visibility = "collapsed",
            key              = f"query_input_{len(st.session_state.chat_history)}",
        )

    with btn_col:
        ask_btn = st.button("Ask →", use_container_width=True)

    # -- Handle query --
    if ask_btn and query.strip():
        with st.spinner("Searching..."):
            try:
                from generation.answerer import ask as pipeline_ask

                answer = pipeline_ask(
                    query      = query.strip(),
                    video_hash = st.session_state.video_hash,
                )

                st.session_state.chat_history.append({
                    "query"  : query.strip(),
                    "answer" : answer,
                })

                st.rerun()

            except Exception as e:
                st.error(f"Query failed: {e}")
                logger.exception("Query error")

    # -- Empty state hint --
    if not st.session_state.chat_history:
        st.markdown("""
<div style="text-align:center; padding: 3rem 0; color:#2a2a2a;">
    <div style="font-family:'Syne',sans-serif; font-size:2rem;
                font-weight:800; letter-spacing:-0.02em;">
        Ask anything.
    </div>
    <div style="font-size:0.75rem; color:#333; margin-top:0.5rem;
                font-family:'DM Mono',monospace;">
        Try: "explain attention mechanism" or "what is gradient descent"
    </div>
</div>
        """, unsafe_allow_html=True)
