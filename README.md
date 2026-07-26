# Amazon Customer Review Analysis 

Welcome to the **Amazon Customer Review Analysis** project! This repository contains a complete end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to process, analyze, and query Amazon customer reviews using a Retrieval-Augmented Generation (RAG) approach.

##  Project Overview

The goal of this project is to extract meaningful insights from customer reviews. By leveraging vector databases and advanced prompting techniques, this application allows users to interact with the review data through a user-friendly web interface.

## 📁 Repository Structure

The project is structured sequentially to demonstrate the complete data pipeline:

* **`1_doc.py`**: Handles initial document loading and parsing.
* **`2_preprocessing.py`**: Cleans and prepares the text data for analysis.
* **`3_chunking.py`**: Splits the preprocessed text into manageable chunks.
* **`4_vector_representation.py`**: Converts text chunks into vector embeddings.
* **`5_create_chroma_store.py`**: Initializes and populates the ChromaDB vector database.
* **`6_retrieve_context.py`**: Manages the retrieval of relevant context based on user queries.
* **`7_prompting.py`**: Structures the prompts sent to the LLM (Large Language Model) for generating responses.
* **`8_streamlit_app.py`**: The main frontend application built with Streamlit for user interaction.
* **`data.zip`**: The compressed dataset containing the Amazon customer reviews used in this project.
* **`requirements.txt`**: Lists all the Python dependencies required to run the project.

## 🛠️ Installation & Setup

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/BasmaMansour99/Amazon-Customer-Review.git](https://github.com/BasmaMansour99/Amazon-Customer-Review.git)
   cd Amazon-Customer-Review
