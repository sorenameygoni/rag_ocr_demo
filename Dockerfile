FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --default-timeout=1000 tensorflow-cpu tf-keras
RUN pip install --default-timeout=1000 opencv-python-headless streamlit llama-index llama-index-llms-groq llama-index-embeddings-huggingface
EXPOSE 8501
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0