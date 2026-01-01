import os

# --- Windows OpenMP fix (must be BEFORE torch/numpy imports) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
from typing import Dict, List, Optional

import streamlit as st
import torch

from src.net import KhmerRNN
from src.utils import decode_spaces


# -------------------------
# Defaults (edit these)
# -------------------------
DEFAULT_CKPT = "checkpoints/khmer_rnn_best_3.pt"
DEFAULT_VOCAB = "checkpoints/vocab.json"
MAX_LEN = 128  # must match training max_length


# -------------------------
# Helpers
# -------------------------
def chunk_text(text: str, max_len: int) -> List[str]:
    return [text[i : i + max_len] for i in range(0, len(text), max_len)]


@st.cache_resource
def load_vocab(vocab_path: str) -> Dict[str, int]:
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(ckpt_path: str, vocab_size: int, device: str) -> KhmerRNN:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # IMPORTANT: must match training hparams
    model = KhmerRNN(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        rnn_type="lstm",
        residual=False,
        use_crf=False
    )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def segment_text(text: str, model: KhmerRNN, vocab: Dict[str, int], device: str) -> str:
    text = text.strip()
    if not text:
        return ""

    unk = vocab.get("<UNK>", 1)
    chunks = chunk_text(text, MAX_LEN)

    outputs: List[str] = []
    for ch in chunks:
        ids = [vocab.get(c, unk) for c in ch]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        logits = model(x)  # (1, T, 2)
        pred = logits.argmax(dim=-1).squeeze(0).tolist()
        outputs.append(decode_spaces(ch, pred))

    return "".join(outputs)


# -------------------------
# Session state init
# -------------------------
def ss_init():
    st.session_state.setdefault("mode", "Paste")          # "Paste" or "Upload"
    st.session_state.setdefault("paste_text", "")
    st.session_state.setdefault("uploaded_text", "")
    st.session_state.setdefault("seg_out", "")
    st.session_state.setdefault("uploader_key", 0)        # change to clear uploader


ss_init()


def clear_all():
    st.session_state["paste_text"] = ""
    st.session_state["uploaded_text"] = ""
    st.session_state["seg_out"] = ""
    st.session_state["mode"] = "Paste"
    st.session_state["uploader_key"] += 1  # forces file_uploader to reset


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Khmer Space Injector", page_icon="📝", layout="wide")
st.title("📝 Khmer Space Injector")
st.caption("Paste Khmer text or upload a .txt file → get segmented output + download.")

with st.sidebar:
    st.header("Model Paths")
    ckpt_path = st.text_input("Checkpoint (.pt)", value=DEFAULT_CKPT)
    vocab_path = st.text_input("Vocab (vocab.json)", value=DEFAULT_VOCAB)

    use_cuda = st.checkbox("Use CUDA if available", value=False)
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    st.write(f"**Device:** {device}")

    st.divider()
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload .txt",
        type=["txt"],
        key=f"uploader_{st.session_state['uploader_key']}",
    )

# If a file is uploaded, auto-switch to Upload mode and store its content
if uploaded_file is not None:
    st.session_state["uploaded_text"] = uploaded_file.read().decode("utf-8", errors="ignore")
    st.session_state["mode"] = "Upload"

# Load resources
try:
    vocab = load_vocab(vocab_path)
    model = load_model(ckpt_path, vocab_size=len(vocab), device=device)
except Exception as e:
    st.error(str(e))
    st.stop()

# Mode selector (acts like tabs but controllable)
st.session_state["mode"] = st.radio(
    "Choose input method",
    options=["Paste", "Upload"],
    index=0 if st.session_state["mode"] == "Paste" else 1,
    horizontal=True,
)

# Input area
if st.session_state["mode"] == "Paste":
    st.text_area(
        "Input text",
        key="paste_text",
        height=220,
        placeholder="បញ្ចូលអត្ថបទខ្មែរនៅទីនេះ...",
    )
    text_to_process = st.session_state["paste_text"]

else:
    st.text_area(
        "Uploaded text preview",
        value=st.session_state["uploaded_text"],
        height=220,
        disabled=True,
    )
    text_to_process = st.session_state["uploaded_text"]

st.divider()

# Buttons
col1, col2 = st.columns([1, 1])
with col1:
    run_btn = st.button("✅ Segment", type="primary")
with col2:
    clear_btn = st.button("🧹 Clear")

if clear_btn:
    clear_all()
    st.rerun()

if run_btn:
    if not text_to_process or not text_to_process.strip():
        st.warning("Please paste text or upload a .txt file first.")
    else:
        st.session_state["seg_out"] = segment_text(text_to_process, model, vocab, device=device)

# Output
st.subheader("✅ Output")
st.text_area("Segmented text", value=st.session_state["seg_out"], height=260)

st.download_button(
    label="⬇️ Download output (.txt)",
    data=st.session_state["seg_out"].encode("utf-8"),
    file_name="segmented_output.txt",
    mime="text/plain",
    disabled=(not st.session_state["seg_out"].strip()),
)
