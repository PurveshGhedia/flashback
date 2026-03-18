# 🎬 Flashback

> Search inside any lecture video using natural language.

Flashback lets you chat with a recorded lecture. Ask a question, get back the exact timestamp, the relevant frame, and a generated explanation — without scrubbing through the timeline.

---

## How It Works

A two-phase RAG pipeline built for lecture video content:

1. **Ingestion** — extracts frames, filters redundancy via adaptive SSIM + CLIP, transcribes audio with Whisper, and indexes everything in ChromaDB with timestamps as metadata.
2. **Retrieval** — embeds your query, runs hybrid search across frames + transcript, re-ranks with Gemini, and returns an answer with exact timestamps and keyframes.

## Stack

`OpenCV` · `CLIP ViT-B/32` · `OpenAI Whisper` · `PySceneDetect` · `ChromaDB` · `sentence-transformers` · `Gemini 1.5 Pro` · `Streamlit`

---

## Getting Started

```bash
git clone https://github.com/your-username/flashback.git
cd flashback
chmod +x environment_setup.sh && ./environment_setup.sh
# Add GEMINI_API_KEY to .env
conda activate lecture-rag
streamlit run app.py
```

---

## Project Structure

```
flashback/
├── app.py
├── config.py
├── ingestion/
│   ├── frame_extractor.py
│   ├── ssim_filter.py
│   ├── clip_filter.py
│   ├── scene_detector.py
│   ├── transcriber.py
│   └── indexer.py
├── retrieval/
│   ├── embedder.py
│   ├── searcher.py
│   └── reranker.py
├── generation/
│   └── answerer.py
└── evaluation/
    ├── annotator.py
    └── metrics.py
```

---

## License

MIT
