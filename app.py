import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer 
from ocr_engine import load_ocr_model, predict_ocr
st.set_page_config(page_title = "AI Lawyer Assistant" , page_icon ="⚖️", layout="wide")
st.title("OCR powered RAG 🤖")
@st.cache_resource
def get_model():
    return load_ocr_model()
try:
    ocr_model = get_model()
except Exception as e :
    st.error(f"error loading ocr:{e}")
    st.stop()
with st.sidebar:
    st.header("Setting")
    api_key = st.text_input("Groq API Key", value="gsk_...", type="password")
    uploaded_file = st.file_uploader("Uploade PDF or document", type=["pdf", "png", "jpg", "jpeg"])
    if st.button("start new chat"):
        st.session_state.messages = [] 
        st.session_state.chat_engine = None
        st.success("Memory clean")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uoloaded_file and api_key:
    os.environ["GROQ_API_KEY"] = api_key
    Settings.llm = Groq(model="llama-3.3-70b-versatile")
    Settings.embed_model = HuggingFaceEmbedding(model = "BAAI/bge-small-en-v1.5")
    
    if st.session_state.chat_engine is None:
        with st.spinner("processing documents"):   
            with tempfile.NamedTemporaryFile(delete = False , suffix = uploaded_file.name as tmp_file:
                tmp_file.write(uploaded_file.get_buffer())	
                tmp_path = tmp_file.name
                
            extracted_text = ""
            try:
                if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                    extracted_text = predict_ocr(ocr_model , tmp_path)
                elif uploaded_file.type == "application/pdf":
                    from llama_index.core import SimpleDirectoryReader
                    docs = SimpleDirectoryReader(input_file = [tmp_path]).load_data()
                for doc in docs:
                        extracted_text += doc
            except Exception as e:
                st.error(f"error in reading file : {e}")
                st.stop()
            if extracted_text :
                documents = [Document(text = extracted_text)]
                index = VectorStoreIndex.from_documents(documents)
                memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
                st.session_state.chat_engine = index.as_chat_engine(chat_mode="context",memory=memory,system_prompt="Just answer according to the provided document if you dont know say : there is no information similar in the documets")
                st.success("document peoccesed")
            else:
                st.warning("nothing found! please check image quality")
    if prompt := st.chat_input("Ask your question"): 
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_messages("user"):
            st.markdown(prompt)
        if st.session_state.chat_engine:
            with st.chat_message("assistant"):
                with st.spinner("proccesing"):
                    response=st.session_state.chat_engine.chat(prompt)
                    sources_text=""
                    for node in response.source_nodes:
                        score = f"{node.score:.2f}" if node.score else "N/A"
                        ref_text = node.node.get_content()[:100].replace("\n", " ") + "..." 
                        sources_text +=	f"-**(precision{score})**:{ref_text}\n"	
                    st.markdown(response.response)
            
                                                 
                    if sources_text:
                        with st.expander("source answer 📚 "):
                            st.markdown(sources_text)
                    
                    st.session_state.messages.append({"role" , "assistant", "content": response.response})
        
elif not api_key:
    st.info("please enter API_KEY")







