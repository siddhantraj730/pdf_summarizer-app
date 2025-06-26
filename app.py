import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    pipeline
)
import torch
import base64
import os

# ---- Streamlit Config ----
st.set_page_config(layout="wide")
st.title("📄 Document Summarization App (Offline or Online Model)")

# ---- Model Paths and Checkpoints ----
LOCAL_FLAN_DIR = "./local_flan_t5_large"
LOCAL_LAMINI_FLAN_DIR = "./local_lamini_flan_t5_248m"
LAMINI_FLAN_CHECKPOINT = "MBZUAI/LaMini-Flan-T5-248M"

# ---- Model Loader Functions ----
@st.cache_resource(show_spinner=True)
def load_flan_t5():
    if not os.path.exists(LOCAL_FLAN_DIR):
        os.makedirs(LOCAL_FLAN_DIR, exist_ok=True)
        checkpoint = "google/flan-t5-large"
        st.info("Downloading google/flan-t5-large model + tokenizer... This will happen only once!")
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=torch.float32
        )
        tokenizer.save_pretrained(LOCAL_FLAN_DIR)
        model.save_pretrained(LOCAL_FLAN_DIR)
        st.success(f"Model and tokenizer saved locally at {LOCAL_FLAN_DIR}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_FLAN_DIR)
        model = T5ForConditionalGeneration.from_pretrained(
            LOCAL_FLAN_DIR,
            device_map="auto",
            torch_dtype=torch.float32
        )
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_lamini_flan_t5():
    if not os.path.exists(LOCAL_LAMINI_FLAN_DIR):
        os.makedirs(LOCAL_LAMINI_FLAN_DIR, exist_ok=True)
        st.info("Downloading MBZUAI/LaMini-Flan-T5-248M model + tokenizer... This will happen only once!")
        tokenizer = T5Tokenizer.from_pretrained(LAMINI_FLAN_CHECKPOINT)
        model = T5ForConditionalGeneration.from_pretrained(
            LAMINI_FLAN_CHECKPOINT,
            device_map="auto",
            torch_dtype=torch.float32
        )
        tokenizer.save_pretrained(LOCAL_LAMINI_FLAN_DIR)
        model.save_pretrained(LOCAL_LAMINI_FLAN_DIR)
        st.success(f"Model and tokenizer saved locally at {LOCAL_LAMINI_FLAN_DIR}")
    else:
        tokenizer = T5Tokenizer.from_pretrained(LOCAL_LAMINI_FLAN_DIR)
        model = T5ForConditionalGeneration.from_pretrained(
            LOCAL_LAMINI_FLAN_DIR,
            device_map="auto",
            torch_dtype=torch.float32
        )
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_bart():
    checkpoint = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    return tokenizer, model

# ---- File Preprocessing ----
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    full_text = ''.join([page.page_content for page in pages])
    total_length = len(full_text)
    target_chunks = 10
    chunk_size = max(300, total_length // target_chunks)
    chunk_overlap = chunk_size // 5
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(pages)
    chunks = [text.page_content for text in texts]
    return chunks

# ---- Summarization Pipelines ----
def get_length_params(length_choice, model_type='flan'):
    if length_choice == "Small":
        min_ratio, max_ratio = 0.1, 0.2
        min_cap, max_cap = 20, 50
    elif length_choice == "Medium":
        min_ratio, max_ratio = 0.2, 0.5
        min_cap, max_cap = 40, 100
    else:  # Large
        min_ratio, max_ratio = 0.4, 0.8
        min_cap, max_cap = 80, 200
    if model_type == 'bart':
        max_cap = min(max_cap, 512)
    return min_ratio, max_ratio, min_cap, max_cap

def offline_summarize_with_model(filepath, length_choice, tokenizer, model):
    input_chunks = file_preprocessing(filepath)
    all_summaries = []
    min_ratio, max_ratio, min_cap, max_cap = get_length_params(length_choice, 'flan')
    for chunk in input_chunks:
        input_len = len(tokenizer.encode(chunk, truncation=True))
        max_len = min(max_cap, int(input_len * max_ratio))
        min_len = max(min_cap, int(input_len * min_ratio))
        pipe_sum = pipeline(
            'summarization',
            model=model,
            tokenizer=tokenizer,
            max_length=max_len,
            min_length=min_len
        )
        result = pipe_sum(chunk, truncation=True)
        all_summaries.append(result[0]['summary_text'])
    final_summary = "\n\n".join(all_summaries)
    return final_summary

def online_summarize(filepath, length_choice):
    tokenizer, model = load_bart()
    input_chunks = file_preprocessing(filepath)
    all_summaries = []
    min_ratio, max_ratio, min_cap, max_cap = get_length_params(length_choice, 'bart')
    for chunk in input_chunks:
        input_len = len(tokenizer.encode(chunk, truncation=True))
        max_len = min(max_cap, int(input_len * max_ratio), 512)
        min_len = max(min_cap, int(input_len * min_ratio))
        pipe_sum = pipeline(
            'summarization',
            model=model,
            tokenizer=tokenizer,
            max_length=max_len,
            min_length=min_len
        )
        result = pipe_sum(chunk, truncation=True)
        all_summaries.append(result[0]['summary_text'])
    final_summary = "\n\n".join(all_summaries)
    return final_summary

# ---- PDF Display ----
@st.cache_data
def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# ---- UI ----
mode_choice = st.radio(
    "Select summarization mode:",
    ("Offline", "Online"),
    index=0,
    help="Offline runs locally after first download. Online uses BART (requires internet unless cached)."
)

offline_model = None
online_model = None
if mode_choice == "Offline":
    offline_model = st.selectbox(
        "Select offline model:",
        ("Flan-T5-Large", "LaMini-Flan-T5-248M"),
        index=0
    )
elif mode_choice == "Online":
    online_model = st.selectbox(
        "Select online model:",
        ("BART-Large-CNN",),
        index=0
    )

summary_length = st.radio(
    "Select summary length:",
    ("Small", "Medium", "Large"),
    index=1,
    help="Small: Shortest summary, Medium: Balanced, Large: Most detailed"
)

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    if st.button("Summarize"):
        col1, col2 = st.columns(2)
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        with col1:
            st.info("Uploaded File")
            displayPDF(filepath)

        with col2:
            with st.spinner("Generating summary..."):
                if mode_choice == "Offline":
                    if offline_model == "Flan-T5-Large":
                        tokenizer, model = load_flan_t5()
                        summary = offline_summarize_with_model(filepath, summary_length, tokenizer, model)
                    elif offline_model == "LaMini-Flan-T5-248M":
                        tokenizer, model = load_lamini_flan_t5()
                        summary = offline_summarize_with_model(filepath, summary_length, tokenizer, model)
                elif mode_choice == "Online":
                    summary = online_summarize(filepath, summary_length)
            st.success("✅ Summarization Complete")
            st.markdown(summary)
            st.download_button(
                label="📥 Download Summary as .txt",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
