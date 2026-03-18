"""
app.py
=======
Flashback — Streamlit UI

Pages:
  library   — shows all indexed videos with thumbnails + unified search
  indexing  — progress while a new video is being processed
  chat      — single video chat with seekable video player
"""

import streamlit as st
import logging
import time
import shutil
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Flashback", page_icon="🎬",
    layout="wide", initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.stApp { background-color: #0a0a0a; color: #e8e8e8; }

.flashback-header { text-align: center; padding: 2rem 0 1.2rem 0; }
.flashback-title {
    font-family: 'Syne', sans-serif; font-size: 3rem;
    font-weight: 800; letter-spacing: -0.03em; color: #f5f5f5; margin: 0;
}
.flashback-title span { color: #ff4d00; }
.flashback-subtitle {
    font-size: 0.75rem; color: #444; letter-spacing: 0.12em;
    text-transform: uppercase; margin-top: 0.5rem;
}
.subtle-divider { border: none; border-top: 1px solid #1a1a1a; margin: 1.2rem 0; }
.section-label {
    font-size: 0.7rem; color: #333; text-transform: uppercase;
    letter-spacing: 0.12em; margin-bottom: 0.6rem;
}

/* Video card */
.video-card {
    background: #0f0f0f; border: 1px solid #1e1e1e;
    border-radius: 8px; overflow: hidden; margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.video-card:hover { border-color: #333; }
.video-card-thumb {
    width: 100%; aspect-ratio: 16/9;
    object-fit: cover; display: block; background: #111;
}
.video-card-thumb-placeholder {
    width: 100%; aspect-ratio: 16/9; background: #111;
    display: flex; align-items: center; justify-content: center;
    color: #222; font-size: 2rem;
}
.video-card-body { padding: 0.8rem 1rem; }
.video-card-name {
    font-family: 'Syne', sans-serif; font-size: 0.9rem;
    font-weight: 700; color: #f5f5f5; margin: 0 0 0.3rem 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.video-card-meta { font-size: 0.68rem; color: #444; display: flex; gap: 0.8rem; }
.video-card-meta strong { color: #ff4d00; }

/* Unified search */
.search-bar {
    background: #111; border: 1px solid #222; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 1.5rem;
}

/* Chat */
.msg-user {
    background: #111; border: 1px solid #222;
    border-radius: 8px 8px 2px 8px; padding: 0.8rem 1rem;
    margin: 0.5rem 0; font-size: 0.9rem; color: #ccc;
    max-width: 80%; margin-left: auto;
}
.msg-assistant {
    background: #0f0f0f; border: 1px solid #1e1e1e;
    border-left: 3px solid #ff4d00;
    border-radius: 2px 8px 8px 8px; padding: 0.8rem 1rem;
    margin: 0.5rem 0; font-size: 0.88rem; color: #ddd;
    line-height: 1.6; max-width: 90%;
}
.ts-pill {
    display: inline-block; background: #1a1a1a; border: 1px solid #ff4d00;
    color: #ff4d00; border-radius: 4px; padding: 0.15rem 0.5rem;
    font-size: 0.72rem; margin-right: 0.3rem; letter-spacing: 0.05em;
    cursor: pointer;
}
.video-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 6px; padding: 0.4rem 0.8rem;
    font-size: 0.75rem; color: #888;
}
.stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 100px; background: #111;
    border: 1px solid #1e1e1e; border-radius: 6px;
    padding: 0.7rem 0.8rem; text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif; font-size: 1.5rem;
    font-weight: 700; color: #ff4d00; display: block;
}
.stat-label { font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em; }

/* Cross-video result card */
.xvideo-card {
    background: #0f0f0f; border: 1px solid #1e1e1e;
    border-left: 3px solid #ff4d00; border-radius: 4px 8px 8px 4px;
    padding: 1rem; margin-bottom: 1rem;
}
.xvideo-title {
    font-family: 'Syne', sans-serif; font-size: 0.85rem;
    font-weight: 700; color: #ff4d00; margin-bottom: 0.5rem;
}

/* Inputs & buttons */
.stTextInput > div > div > input {
    background: #111 !important; border: 1px solid #222 !important;
    border-radius: 6px !important; color: #e8e8e8 !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff4d00 !important; box-shadow: none !important;
}
.stButton > button {
    background: #ff4d00 !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important; letter-spacing: 0.05em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
[data-testid="stFileUploader"] {
    background: #111 !important; border: 1px dashed #222 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Logging & session state
# ---------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

defaults = {
    "page": "library",
    "active_video_hash": None,
    "active_video_info": None,
    "chat_history": [],
    "unified_results": None,
    "unified_query": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{h}:{mins:02d}:{secs:02d}" if h > 0 else f"{mins:02d}:{secs:02d}"


def ts_to_seconds(ts_str: str) -> int:
    """Convert MM:SS or H:MM:SS to total seconds."""
    parts = ts_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def save_uploaded_video(uploaded_file) -> str:
    """Save uploaded file to data/videos/ AND static/videos/ for serving."""
    from config import VIDEOS_DIR, STATIC_VIDEOS_DIR
    # Save to data/videos/ (source of truth)
    save_path = VIDEOS_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # Copy to static/videos/ so Streamlit can serve it
    static_path = STATIC_VIDEOS_DIR / uploaded_file.name
    if not static_path.exists():
        shutil.copy2(str(save_path), str(static_path))
    return str(save_path)


def get_static_video_url(video_path: str) -> str:
    """Return the app/static URL for a video file."""
    filename = Path(video_path).name
    return f"app/static/videos/{filename}"


def go_to_chat(video_hash: str, video_info: dict):
    st.session_state.active_video_hash = video_hash
    st.session_state.active_video_info = video_info
    st.session_state.chat_history = []
    st.session_state.page = "chat"


def go_to_library():
    st.session_state.page = "library"
    st.session_state.active_video_hash = None
    st.session_state.active_video_info = None
    st.session_state.chat_history = []
    st.session_state.unified_results = None
    st.session_state.unified_query = ""


@st.cache_resource(show_spinner=False)
def run_indexing(video_path: str, force: bool = False):
    from ingestion.indexer import index_video
    return index_video(video_path, force_reindex=force)


def render_answer(answer, show_player: bool = False,
                  video_path: str = "", start_sec: float = 0):
    """Render answer text, timestamps, keyframes, and optional video player."""
    st.markdown(
        f'<div class="msg-assistant">{answer.response_text}</div>',
        unsafe_allow_html=True
    )
    if answer.timestamps:
        ts_html = " ".join(
            f'<span class="ts-pill">▶ {fmt_ts(t)}</span>'
            for t in answer.timestamps
        )
        st.markdown(f'<div style="margin:0.5rem 0;">{ts_html}</div>',
                    unsafe_allow_html=True)

    # Keyframes
    if answer.keyframe_paths:
        valid = [p for p in answer.keyframe_paths if Path(p).exists()]
        if valid:
            cols = st.columns(min(len(valid), 3))
            for i, (col, path) in enumerate(zip(cols, valid[:3])):
                with col:
                    try:
                        img = Image.open(path)
                        ts = answer.timestamps[i] if i < len(
                            answer.timestamps) else 0
                        st.image(
                            img, caption=f"▶ {fmt_ts(ts)}", use_container_width=True)
                    except Exception:
                        pass

    # Video player — seeks to first relevant timestamp
    if show_player and video_path and answer.timestamps:
        seek_to = int(answer.timestamps[0])
        static_url = get_static_video_url(video_path)

        st.markdown(
            f'<div class="section-label" style="margin-top:1rem;">'
            f'Video at ▶ {fmt_ts(answer.timestamps[0])}</div>',
            unsafe_allow_html=True
        )
        # HTML5 video player with autostart at timestamp
        st.components.v1.html(f"""
        <video
            controls
            autoplay
            style="width:100%; border-radius:6px; background:#000;
                   border:1px solid #1e1e1e; max-height:360px;"
            src="/{static_url}#t={seek_to}">
            Your browser does not support HTML5 video.
        </video>
        """, height=380)


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown("""
<div class="flashback-header">
    <div class="flashback-title">flash<span>back</span></div>
    <div class="flashback-subtitle">search inside any lecture · natural language</div>
</div>
<hr class="subtle-divider">
""", unsafe_allow_html=True)


# ===============================================================
# PAGE — LIBRARY
# ===============================================================
if st.session_state.page == "library":

    from ingestion.indexer import list_indexed_videos, update_video_name, get_display_name

    indexed_videos = list_indexed_videos()

    # -- Unified search bar --
    st.markdown('<div class="section-label">Search across all lectures</div>',
                unsafe_allow_html=True)

    search_col, btn_col = st.columns([5, 1], gap="small")
    with search_col:
        unified_query = st.text_input(
            label="Unified search",
            placeholder="e.g. explain backpropagation, what is attention...",
            label_visibility="collapsed",
            value=st.session_state.unified_query,
            key="unified_input",
        )
    with btn_col:
        search_btn = st.button("Search →", use_container_width=True)

    if search_btn and unified_query.strip():
        st.session_state.unified_query = unified_query.strip()
        with st.spinner("Searching all lectures..."):
            try:
                from generation.answerer import ask_across_videos
                st.session_state.unified_results = ask_across_videos(
                    unified_query.strip()
                )
            except Exception as e:
                st.error(f"Search failed: {e}")
                logger.exception("Unified search error")

    # -- Unified search results --
    if st.session_state.unified_results:
        results = st.session_state.unified_results
        st.markdown(
            f'<div class="section-label" style="margin-top:0.5rem;">'
            f'Found in {len(results)} lecture(s) — '
            f'"{st.session_state.unified_query}"</div>',
            unsafe_allow_html=True
        )

        for vr in results:
            info = vr["video_info"]
            name = get_display_name(info)
            answer = vr["answer"]
            score = vr["best_score"]

            with st.expander(f"▶  {name}  (relevance: {score:.2f})", expanded=True):
                render_answer(
                    answer,
                    show_player=True,
                    video_path=info.get("video_path", ""),
                    start_sec=answer.timestamps[0] if answer.timestamps else 0,
                )
                if st.button(f"Chat with this lecture →",
                             key=f"goto_{info['video_hash']}"):
                    go_to_chat(info["video_hash"], info)
                    st.rerun()

        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Library grid --
    st.markdown('<div class="section-label">Your lecture library</div>',
                unsafe_allow_html=True)

    if not indexed_videos:
        st.markdown("""
        <div style="padding:2rem; background:#0f0f0f; border:1px dashed #1e1e1e;
                    border-radius:8px; text-align:center; color:#333; margin-bottom:1rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;">
                No videos indexed yet</div>
            <div style="font-size:0.75rem; margin-top:0.3rem;">
                Upload a lecture to get started</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 3-column grid
        cols = st.columns(3, gap="medium")
        for i, video in enumerate(indexed_videos):
            with cols[i % 3]:
                hash_ = video["video_hash"]
                name = get_display_name(video)
                kf = video["total_keyframes"]
                chunks = video["total_chunks"]
                date = video.get("date_indexed", "")[:10]
                thumb = video.get("thumbnail_path", "")

                # Thumbnail
                if thumb and Path(thumb).exists():
                    try:
                        img = Image.open(thumb)
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.markdown(
                            '<div class="video-card-thumb-placeholder">🎬</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        '<div style="aspect-ratio:16/9; background:#111; '
                        'border-radius:6px 6px 0 0; display:flex; '
                        'align-items:center; justify-content:center; '
                        'font-size:2rem; color:#222;">🎬</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(f"""
                <div style="padding:0.6rem 0 0.3rem 0;">
                    <div class="video-card-name" title="{name}">{name}</div>
                    <div class="video-card-meta">
                        <span><strong>{kf}</strong> keyframes</span>
                        <span><strong>{chunks}</strong> chunks</span>
                        <span>{date}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Inline name editor
                with st.expander("✏ Rename", expanded=False):
                    new_name = st.text_input(
                        "Name", value=name,
                        key=f"rename_{hash_}",
                        label_visibility="collapsed",
                    )
                    if st.button("Save", key=f"save_name_{hash_}"):
                        update_video_name(hash_, new_name)
                        st.success("Saved!")
                        st.rerun()

                if st.button("Ask →", key=f"chat_{hash_}", use_container_width=True):
                    go_to_chat(hash_, video)
                    st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Upload new video --
    st.markdown('<div class="section-label">Index a new video</div>',
                unsafe_allow_html=True)

    up_col, info_col = st.columns([2, 1], gap="large")

    with up_col:
        uploaded = st.file_uploader(
            label="Upload", type=["mp4", "mov", "avi", "mkv", "webm"],
            label_visibility="collapsed",
        )

        custom_name = ""
        if uploaded:
            st.markdown(
                f'<div class="video-badge" style="margin-bottom:0.8rem;">'
                f'🎬 {uploaded.name} · {uploaded.size / 1e6:.1f} MB</div>',
                unsafe_allow_html=True
            )
            custom_name = st.text_input(
                "Lecture name",
                placeholder="e.g. CS224N Lecture 1 — Introduction to NLP",
                help="Give this lecture a memorable name. You can rename it later.",
            )
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Index Video →", use_container_width=False):
                with st.spinner("Saving..."):
                    video_path = save_uploaded_video(uploaded)

                progress = st.progress(0)
                status = st.empty()
                steps = [
                    (0.05, "Detecting scenes..."),
                    (0.20, "Extracting frames..."),
                    (0.40, "SSIM filtering..."),
                    (0.55, "CLIP filtering..."),
                    (0.75, "Whisper transcription..."),
                    (0.92, "Indexing to ChromaDB..."),
                ]
                for pct, msg in steps:
                    progress.progress(pct)
                    status.markdown(
                        f'<span style="font-size:0.78rem; color:#666;">{msg}</span>',
                        unsafe_allow_html=True
                    )
                    if pct < 0.40:
                        time.sleep(0.3)

                try:
                    result = run_indexing(video_path)

                    # Save custom name to registry
                    if custom_name.strip():
                        from ingestion.indexer import update_video_name
                        update_video_name(result.video_hash,
                                          custom_name.strip())

                    progress.progress(1.0)
                    status.markdown(
                        '<span style="font-size:0.78rem; color:#ff4d00;">✓ Indexed</span>',
                        unsafe_allow_html=True
                    )
                    time.sleep(0.6)

                    video_info = {
                        "video_hash": result.video_hash,
                        "video_name": uploaded.name,
                        "custom_name": custom_name.strip(),
                        "video_path": video_path,
                        "total_keyframes": result.total_keyframes,
                        "total_chunks": result.total_chunks,
                        "ssim_reduction_pct": result.ssim_reduction_pct,
                        "clip_reduction_pct": result.clip_reduction_pct,
                        "thumbnail_path": result.thumbnail_path,
                    }
                    go_to_chat(result.video_hash, video_info)
                    st.rerun()

                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.exception("Indexing error")

    with info_col:
        st.markdown("""
        <div style="padding:1.2rem; background:#0f0f0f; border:1px solid #1a1a1a;
                    border-radius:8px; margin-top:1.8rem;">
            <p style="font-family:'Syne',sans-serif; font-size:0.85rem;
                      font-weight:700; color:#f5f5f5; margin:0 0 0.8rem 0;">Tips</p>
            <div style="font-size:0.75rem; color:#444; line-height:1.8;">
                Give lectures descriptive names like<br>
                <span style="color:#666;">"CS224N L1 — Intro to NLP"</span><br>
                so they're easy to find later.<br><br>
                Use unified search to find content<br>
                across all your lectures at once.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===============================================================
# PAGE — CHAT
# ===============================================================
elif st.session_state.page == "chat":

    from ingestion.indexer import get_display_name

    info = st.session_state.active_video_info or {}
    hash_ = st.session_state.active_video_hash
    name = get_display_name(info) if info else hash_

    # -- Top bar --
    top_left, top_right = st.columns([3, 1], gap="small")
    with top_left:
        st.markdown(
            f'<div class="video-badge">🎬 {name} '
            f'· {info.get("total_keyframes", "?")} keyframes '
            f'· {info.get("total_chunks", "?")} chunks</div>',
            unsafe_allow_html=True
        )
    with top_right:
        if st.button("← Library", use_container_width=True):
            go_to_library()
            st.rerun()

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Stats --
    ssim_d = ("N/A" if info.get("ssim_reduction_pct", -1) < 0
              else f"{info.get('ssim_reduction_pct', 0):.0f}%")
    clip_d = ("N/A" if info.get("clip_reduction_pct", -1) < 0
              else f"{info.get('clip_reduction_pct', 0):.0f}%")

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <span class="stat-value">{info.get("total_keyframes", "?")}</span>
            <div class="stat-label">Keyframes</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{info.get("total_chunks", "?")}</span>
            <div class="stat-label">Transcript chunks</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{ssim_d}</span>
            <div class="stat-label">SSIM reduction</div>
        </div>
        <div class="stat-card">
            <span class="stat-value">{clip_d}</span>
            <div class="stat-label">CLIP reduction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Chat history --
    video_path = info.get("video_path", "")

    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            st.markdown(
                f'<div class="msg-user">{entry["query"]}</div>',
                unsafe_allow_html=True
            )
            if entry.get("answer"):
                render_answer(
                    entry["answer"],
                    show_player=True,
                    video_path=video_path,
                    start_sec=entry["answer"].timestamps[0]
                    if entry["answer"].timestamps else 0,
                )
        st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # -- Query input --
    st.markdown('<div class="section-label">Ask about this lecture</div>',
                unsafe_allow_html=True)

    input_col, btn_col = st.columns([5, 1], gap="small")
    with input_col:
        query = st.text_input(
            label="Query",
            placeholder="e.g. explain attention mechanism, what is word2vec...",
            label_visibility="collapsed",
            key=f"q_{len(st.session_state.chat_history)}",
        )
    with btn_col:
        ask_btn = st.button("Ask →", use_container_width=True)

    if ask_btn and query.strip():
        with st.spinner("Searching..."):
            try:
                from generation.answerer import ask as pipeline_ask
                answer = pipeline_ask(query=query.strip(), video_hash=hash_)
                st.session_state.chat_history.append({
                    "query": query.strip(),
                    "answer": answer,
                })
                st.rerun()
            except Exception as e:
                st.error(f"Query failed: {e}")
                logger.exception("Query error")

    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align:center; padding:3rem 0; color:#2a2a2a;">
            <div style="font-family:'Syne',sans-serif; font-size:1.8rem;
                        font-weight:800;">Ask anything.</div>
            <div style="font-size:0.75rem; margin-top:0.4rem;">
                Try: "explain self-attention" or "what are RNN limitations"</div>
        </div>
        """, unsafe_allow_html=True)
