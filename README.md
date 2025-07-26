# Danica Pension RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about Danica Pension. It scrapes content from the Danica Pension website and PDF documents, indexes the information, and uses a Large Language Model (LLM) to provide answers based on the retrieved content.

## Features

- **Web and PDF Scraping**: Dynamically scrapes content from specified URLs, including both web pages and PDF files.
- **Parent Document Retriever**: Uses a parent-document retriever strategy. It splits documents into larger parent chunks and smaller child chunks. The child chunks are used for embedding and retrieval, but the full parent chunk is returned for better context.
- **Customizable RAG Pipeline**: The `RAG` class allows for easy configuration of chunk sizes, overlaps, and the number of documents to retrieve.
- **Custom Prompts**: The system uses a custom prompt, instructing the model to answer in Danish and to be truthful to the provided context.
- **Persistent Vector Store**: Utilizes ChromaDB to store document embeddings, allowing for persistence between sessions.

## How It Works

The system is composed of two main Python files:

### 1. `scrape_danica.py`

This script is responsible for fetching the data.

- `scrape_web_page(url)`: Takes a URL, fetches the HTML content, and uses BeautifulSoup to extract all text, returning it as a LangChain `Document`.
- `scrape_pdf(url)`: Takes a URL to a PDF, downloads it, and uses `PyPDFLoader` to load its text content into LangChain `Document` objects.
- `scrape_pages(urls)`: Orchestrates the scraping process. It takes a list of URLs, determines whether each is a web page or a PDF, and calls the appropriate function. It returns a list of all scraped `Document` objects.

### 2. `retrieval_augmented_generation.py`

This script defines the core `RAG` class that powers the question-answering system.

- **`__init__(...)`**: Initializes the RAG system with parameters for chunking (`parent_chunk_size`, `child_chunk_size`, etc.) and retrieval (`k`).
- **`setup(documents)`**: This is the main setup method:
    1.  **Initializes Splitters**: Sets up `RecursiveCharacterTextSplitter` for both parent and child documents.
    2.  **Initializes Vector Store**: Uses `Chroma` with `OpenAIEmbeddings` to store the child document embeddings. The vector database is persisted in the `./chroma_db_openai_parent_child/` directory.
    3.  **Initializes Retriever**: Sets up a `ParentDocumentRetriever` which links the smaller, retrieved child documents back to their larger parent documents.
    4.  **Adds Documents**: The scraped documents are processed by the retriever and stored.
    5.  **Creates QA Chain**: It builds a `RetrievalQA` chain, combining the retriever with an `OpenAI` LLM (`gpt-4.1`). It injects a custom Danish prompt to guide the model's responses.
- **`query(question)`**: Takes a user's question, invokes the QA chain, and prints the answer along with the source documents that were used to generate it.

### 3. `app.py`

This script provides a simple web interface for the RAG chatbot using Streamlit.

- It initializes the `RAG` system.
- It provides a text input for the user to ask questions.
- It displays the answer returned by the RAG system.
- It caches the RAG system to avoid reloading the model on each interaction.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY='your_openai_api_key'
    ```

## Web Interface

This project includes a Streamlit application to interact with the RAG model through a user-friendly web interface.

To run the web app, execute the following command in your terminal:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your browser. You can then enter your questions about Danica Pension and receive answers from the chatbot.

## Programmatic Usage

The system can also be used from a script or a Jupyter Notebook (like `test.ipynb`).

1.  **Import the necessary classes:**
    ```python
    from retrieval_augmented_generation import RAG
    from scrape_danica import scrape_pages
    ```

2.  **Define the data sources:**
    ```python
    urls = [
        "https://danica.dk/privat/din-pension/dine-investeringsmuligheder/alle-investeringsprodukter/danica-balance",
        "https://danica.dk/om-danica-pension/kort-om-os",
        "https://danica.dk/-/media/pdf/danica-pension/dk/forsikringsbetingelser/daekning-ved-sundhedssikring.pdf"
    ]
    ```

3.  **Scrape the content:**
    ```python
    documents = scrape_pages(urls)
    ```

4.  **Initialize and set up the RAG system:**
    ```python
    rag_system = RAG(
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=400,
        child_chunk_overlap=50,
        k=10
    )
    rag_system.setup(documents=documents)
    ```

5.  **Ask a question:**
    ```python
    result = rag_system.query("Hvad dækker Danica sundhedspakke?")
    ```
