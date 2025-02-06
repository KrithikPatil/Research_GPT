import os
# Set this before importing other libraries to suppress OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import faiss
import json
import numpy as np
import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler('app.log')      # Log to file
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def load_resources():
    """
    Load all required resources with error handling
    """
    try:
        # Load the DistilBERT model and tokenizer
        MODEL_PATH = "./distilbert_model"
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

        # Load the FAISS index
        FAISS_INDEX_PATH = "./faiss_index.index"
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)

        # Load the dataset
        DATA_PATH = "processed_springer_papers_DL.json"
        with open(DATA_PATH, "r", encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        return model, tokenizer, faiss_index, df

    except Exception as e:
        logger.error(f"Error loading resources: {e}")
        raise

# Load resources when the module is imported
try:
    MODEL, TOKENIZER, FAISS_INDEX, DATAFRAME = load_resources()
except Exception as e:
    logger.error("Failed to load resources. Server cannot start.")
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request
        if not request.is_json:
            logger.warning("Non-JSON request received")
            return jsonify({"error": "Content-Type must be application/json"}), 400

        # Parse JSON data
        data = request.get_json()
        if not isinstance(data, dict):
            logger.warning("Invalid JSON format")
            return jsonify({"error": "Invalid JSON format"}), 400

        # Validate query
        query = data.get("query", "").strip()
        if not query:
            logger.warning("Empty query received")
            return jsonify({"error": "Query cannot be empty"}), 400

        logger.info(f"Processing query: {query}")

        # Generate embedding
        inputs = TOKENIZER(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = MODEL(**inputs, output_hidden_states=True)
            query_embedding = outputs.hidden_states[-1].mean(dim=1).numpy()

        # Similarity search
        k = min(5, len(DATAFRAME))
        distances, indices = FAISS_INDEX.search(query_embedding.astype(np.float32), k)

        # Retrieve relevant papers
        relevant_papers = []
        for i in range(k):
            index = indices[0][i]
            if 0 <= index < len(DATAFRAME):
                paper = DATAFRAME.iloc[index]
                relevant_papers.append({
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "abstract": paper["cleaned_abstract"],
                    "similarity_score": float(distances[0][i])
                })

        logger.info(f"Found {len(relevant_papers)} relevant papers")

        return jsonify({
            "success": True,
            "query": query,
            "results": relevant_papers
        })

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({
            "error": "Internal server error", 
            "details": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)