import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
import os


def scrape_web_page(url):
    """
    Fetches the raw text of a webpage.
    
    Args:
        url (str): The URL of the webpage to scrape.
    """
    
    # Send a GET request to the webpage
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the page
    text = soup.get_text(separator='\n', strip=True)

    return Document(page_content=text, metadata={"source": url})


def scrape_pdf(url):
    """
    Fetches the raw text of a PDF file.
    
    Args:
        url (str): The URL of the PDF file to scrape.
    
    Returns:
        Document: A Document object containing the PDF content.
    """
    
    # Download the PDF file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Save the PDF to a temporary file
    with open('temp.pdf', 'wb') as f:
        f.write(response.content)

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader('temp.pdf')
    documents = loader.load()

    # Clean up the temporary file
    os.remove('temp.pdf')

    return documents if documents else [Document(page_content="", metadata={"source": url})]


def scrape_pages(urls):
    """
    Scrapes multiple webpages / PDFs and returns their content as a list of Document objects.
    
    Args:
        urls (list): A list of URLs to scrape.
    
    Returns:
        list: A list of Document objects containing the scraped content.
    """
    
    documents = []
    for url in urls:
        try:
            if url.lower().endswith('.pdf'):
                docs = scrape_pdf(url)
                if isinstance(docs, list):
                    documents.extend(docs)
                else:
                    documents.append(docs)
            else:
                doc = scrape_web_page(url)
                documents.append(doc)
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return documents