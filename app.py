import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from ocr_engine import load_ocr_model, predict_ocr

st.set_page_config(page_title="Smart Invoice Chat", page_icon="🧠")
st.title("AI Document Assistant (OCR + RAG)")


@st.cache_resource
def get_model():
   
    return load_ocr_model("best_ocr_model.weights.h5")

try:
    ocr_model = get_model()
    st.sidebar.success("OCR Model Loaded! ✅")
except Exception as e:
    st.sidebar.error(f"Error loading OCR model: {e}")
    st.stop()


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key", value="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", type="password")
    uploaded_file = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])


if uploaded_file is not None and api_key:
    
   
    os.environ["GROQ_API_KEY"] = api_key
    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    extracted_text = ""
    
    with st.spinner("Processing File..."):
     
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

       
        
        if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            st.info("Reading text with Custom OCR Model...")
            try:
               
                extracted_text = predict_ocr(ocr_model, tmp_path)
                st.text_area("OCR Output:", extracted_text)
            except Exception as e:
                st.error(f"OCR Failed: {e}")

        
        elif uploaded_file.type == "application/pdf":
            from llama_index.core import SimpleDirectoryReader
            docs = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
            for doc in docs:
                extracted_text += doc.text


    if extracted_text and len(extracted_text.strip()) > 0:
        documents = [Document(text=extracted_text)]
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        st.success("Analysis Complete! Ask me anything.")
        
        user_question = st.text_input("Question about this document:")
        if user_question:
            with st.spinner("Thinking..."):
                response = query_engine.query(user_question)
                st.markdown(f"### Answer:\n{response}")
    else:
        st.warning("Could not extract any text. Please check the image quality.")

elif not api_key:
    st.warning("Please enter API Key")
else:
    st.info("Please upload a file to start.")