# My RAG (Retrieval-Augmented Generation) Attempt

This project is a Streamlit-based application that implements a Retrieval-Augmented Generation (RAG) system. It allows users to upload PDF documents (such as CVs), processes the text, and then performs question-answering using a conversational AI model.

## Features

- **PDF Upload**: Users can upload multiple PDF files.
- **Text Processing**: Extracts text from the uploaded PDFs and splits it into manageable chunks.
- **Vector Store Creation**: Converts text chunks into vector embeddings using HuggingFace's Instruct Embeddings.
- **Conversational AI**: Implements a conversation chain that retrieves relevant information from the text chunks and generates responses using a pre-trained model hosted on HuggingFace.
- **Memory Management**: The application maintains conversational context using a buffer memory.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>


### Instructions:

- **Ensure that the `.env` file contains any API keys** needed for models from HuggingFace or OpenAI.
