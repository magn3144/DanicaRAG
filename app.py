import streamlit as st
from retrieval_augmented_generation import RAG
from scrape_danica import scrape_pages

# --- Constants ---
PARENT_CHUNK_SIZE = 20000
PARENT_CHUNK_OVERLAP = 2000
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 200
K = 50
URLS = [
    "https://danica.dk/privat/din-pension/dine-investeringsmuligheder/alle-investeringsprodukter/danica-balance",
    "https://danica.dk/om-danica-pension/kort-om-os",
    "https://danica.dk/privat/find-hjaelp/pensionsopsparing-og-indbetaling/overfoerelse-af-pensionopsparing#content-list-0-item-0",
    "https://danica.dk/-/media/pdf/danica-pension/dk/forsikringsbetingelser/daekning-ved-sundhedssikring.pdf"
]

# --- App Title ---
st.title("Danica Pension Chatbot")
st.write("Stil et spørgsmål om din pension hos Danica, og få svar baseret på deres officielle dokumentation.")

# --- Model Loading ---
@st.cache_resource
def load_rag_system():
    """
    Loads and initializes the RAG system.
    This function is cached to avoid reloading the model on every run.
    """
    with st.spinner("Initialiserer RAG-systemet... Dette kan tage et øjeblik."):
        rag_system = RAG(
            parent_chunk_size=PARENT_CHUNK_SIZE,
            parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
            child_chunk_size=CHILD_CHUNK_SIZE,
            child_chunk_overlap=CHILD_CHUNK_OVERLAP,
            k=K
        )
        
        # Scrape pages and setup the RAG system
        documents = scrape_pages(URLS)
        rag_system.setup(documents=documents)
    
    return rag_system

rag_system = load_rag_system()

# --- User Interaction ---
question = st.text_input("Indtast dit spørgsmål her:")

if st.button("Stil spørgsmål"):
    if question:
        with st.spinner("Søger efter svar..."):
            answer = rag_system.query(question)
            st.success("Færdig!")
            st.write(answer)
    else:
        st.warning("Indtast venligst et spørgsmål.")
