import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ParentDocumentRetriever
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv


class RAG:
    """
    A class to implement a Retrieval-Augmented Generation (RAG) system.
    This class handles the setup of the RAG pipeline, including document splitting,
    embedding, vector storage, and the creation of a question-answering chain.
    """
    def __init__(self, parent_chunk_size: int, parent_chunk_overlap: int, child_chunk_size: int, child_chunk_overlap: int, k: int) -> None:
        """
        Initializes the RAG class with configuration parameters for document processing.

        Args:
            parent_chunk_size (int): The character size of the parent chunks.
            parent_chunk_overlap (int): The character overlap between parent chunks.
            child_chunk_size (int): The character size of the child chunks.
            child_chunk_overlap (int): The character overlap between child chunks.
            k (int): The number of documents to retrieve for context.
        """
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.k = k
        self.qa_chain = None

    def setup(self, documents: list, verbose: bool = False) -> None:
        """
        Sets up the RAG system by initializing the necessary components.

        This method configures the document splitters, vector store, retriever,
        and the question-answering chain with a custom prompt.

        Args:
            documents (list): A list of Document objects to be indexed.
        """

        # Load environment variables
        load_dotenv()

        # Initialize splitters
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size, chunk_overlap=self.parent_chunk_overlap)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size, chunk_overlap=self.child_chunk_overlap)

        # Initialize stores
        doc_store = InMemoryStore()
        embeddings = OpenAIEmbeddings()

        # Initialize vector store for OpenAI
        vector_store = Chroma(
            collection_name="parent_document_chunks_openai",
            embedding_function=embeddings,
            persist_directory="./chroma_db_openai_parent_child"
        )
        if not os.path.exists("./chroma_db_openai_parent_child"):
            os.makedirs("./chroma_db_openai_parent_child")

        # Initialize the retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=doc_store,
            parent_splitter=parent_splitter,
            child_splitter=child_splitter,
            search_kwargs={"k": self.k}
        )

        if verbose:
            print("Adding documents to the retriever...")
        retriever.add_documents(documents, ids=None)
        if verbose:
            print(f"Retriever indexed {len(vector_store.get()['ids'])} child chunks and {len(doc_store.store)} parent documents.")

        # Initialize the LLM with an OpenAI model
        llm = ChatOpenAI(model="gpt-4.1", temperature=0.4)

        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Brug følgende stykker kontekst til at besvare spørgsmålet til sidst.
            Hvis du ikke kender svaret, så sig blot, at du ikke ved det - prøv ikke at finde på et svar.
            Svar altid på dansk.
            {context}

            Question: {question}
            Answer:
            """
        )

        # Create a RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt},
        )
    
    def query(self, question: str, verbose: bool = False) -> str:
        """
        Executes a query against the RAG system.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The result from the QA chain, containing the answer and source documents.
        """
        if self.qa_chain is None:
            raise RuntimeError("RAG system is not set up. Please call setup() before querying.")
        result = self.qa_chain.invoke({"query": question})

        if verbose:
            print(f"\n--- Query: {question} ---")
            print(f"\n--- Answer: {result['result']} ---")

            if 'source_documents' in result:
                print("\n--- Source Documents Used (Parent Documents): ---")
                for i, doc in enumerate(result['source_documents']):
                    # Note: The 'page' metadata will come from the parent document if preserved by the loader
                    print(f"Parent Document {i+1} (Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}\n----------------------\n")

        return result['result']