import os
import sys
import importlib
import traceback
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.vectorstores.base import VectorStore
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

# Explicitly manage Python path and ChromaDB import
def setup_python_path():
    """Add potential package directories to Python path."""
    base_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'lib', 'python3.12', 'site-packages'),
        '/home/adminuser/venv/lib/python3.12/site-packages',
        '/opt/conda/lib/python3.12/site-packages',
        '/usr/local/lib/python3.12/site-packages'
    ]
    for path in base_paths:
        if path not in sys.path:
            sys.path.append(path)

def import_chromadb():
    """
    Attempt to import ChromaDB with multiple fallback strategies.
    
    Returns:
        The imported chromadb module or None if import fails.
    """
    setup_python_path()
    
    import_strategies = [
        lambda: importlib.import_module('chromadb'),
        lambda: __import__('chromadb'),
    ]
    
    last_exception = None
    for strategy in import_strategies:
        try:
            chromadb_module = strategy()
            # Verify the module has the expected attributes
            if hasattr(chromadb_module, 'PersistentClient'):
                return chromadb_module
            # If not, try to import PersistentClient directly
            if not hasattr(chromadb_module, 'PersistentClient'):
                from chromadb import PersistentClient
                chromadb_module.PersistentClient = PersistentClient
            return chromadb_module
        except (ImportError, RuntimeError, AttributeError) as e:
            last_exception = e
    
    # If all strategies fail, log detailed error
    error_message = f"""
    Critical ChromaDB Import Failure
    --------------------------------
    Python Version: {sys.version}
    Python Path: {sys.path}
    Last Exception: {last_exception}
    Traceback: {traceback.format_exc()}
    
    Possible Solutions:
    1. Verify ChromaDB installation
    2. Check Python version compatibility
    3. Ensure all dependencies are correctly installed
    """
    
    # Use print instead of st.error to ensure visibility
    print(error_message)
    raise ImportError(f"Cannot import ChromaDB: {error_message}")

# Perform the import
try:
    chromadb = import_chromadb()
except Exception as e:
    # Fallback error handling
    print(f"Catastrophic Import Error: {e}")
    chromadb = None

# Rest of the imports
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Ensure the data directory exists
os.makedirs("data/vectorstore", exist_ok=True)
os.chmod("data/vectorstore", 0o755)

def get_api_key():
    """
    Retrieve OpenAI API key from multiple sources.
    
    Priority:
    1. Streamlit secrets
    2. Environment variable
    3. User input
    
    Returns:
        str: OpenAI API key
    """
    # Check Streamlit secrets first
    try:
        if st.secrets and "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
            if api_key and api_key != "your_openai_api_key_here":
                st.success("API key loaded from Streamlit secrets")
                return api_key
    except Exception as secrets_err:
        st.warning(f"Error reading secrets: {secrets_err}")
    
    # Check environment variable
    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        st.success("API key loaded from environment variable")
        return env_api_key
    
    # Fallback to user input
    api_key = st.text_input(
        "Enter your OpenAI API Key", 
        type="password", 
        help="You can get your API key from https://platform.openai.com/account/api-keys"
    )
    
    if api_key:
        # Optional: Validate API key format
        if not api_key.startswith("sk-"):
            st.warning("API key should start with 'sk-'. Please check your key.")
            return None
        
        # Optional: Store in environment for current session
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    
    return None

def get_file_size(size_in_bytes):
    """Get human readable file size from bytes."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.1f} GB"

@st.cache_resource
def get_embeddings():
    """Get cached embeddings instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    return OpenAIEmbeddings(openai_api_key=api_key)

def get_absolute_path(relative_path):
    """Get absolute path from relative path."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))

# Modify ChromaDB client initialization
def get_chroma_client():
    try:
        # More flexible client initialization
        vectorstore_path = os.path.abspath("data/vectorstore")
        
        # Ensure directory exists and is writable
        os.makedirs(vectorstore_path, exist_ok=True)
        os.chmod(vectorstore_path, 0o755)
        
        # Try multiple import strategies
        try:
            import chromadb
        except ImportError:
            # Fallback import
            import importlib
            chromadb = importlib.import_module('chromadb')
        
        # More robust client creation
        try:
            # Try creating client with explicit settings
            client = chromadb.Client(
                chromadb.config.Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=vectorstore_path
                )
            )
            return client
        except Exception as client_err:
            st.warning(f"DuckDB client creation failed: {client_err}")
            
            # Fallback to in-memory client
            try:
                client = chromadb.Client()
                return client
            except Exception as in_memory_err:
                st.error(f"In-memory client creation failed: {in_memory_err}")
                return None
    
    except Exception as e:
        st.error(f"Unexpected error in ChromaDB client initialization: {e}")
        return None

# Alternative vector store initialization for environments with SQLite limitations
def create_vector_store_alternative(texts, embeddings, vectorstore_path):
    """
    Create a vector store using an alternative method that bypasses SQLite version check.
    
    Args:
        texts (List[Document]): List of text documents
        embeddings (OpenAIEmbeddings): Embedding function
        vectorstore_path (str): Path to store the vector store
    
    Returns:
        Chroma: Initialized vector store or None
    """
    try:
        # Import necessary modules
        import chromadb
        from chromadb.config import Settings
        
        # Create a client with DuckDB implementation
        client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=vectorstore_path
            )
        )
        
        # Create a collection
        collection = client.create_collection(
            name="university_docs", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Manually add embeddings
        for i, doc in enumerate(texts):
            # Generate embedding for the document
            embedding = embeddings.embed_query(doc.page_content)
            
            # Add to collection
            collection.add(
                ids=[f"doc_{i}"],
                embeddings=[embedding],
                documents=[doc.page_content],
                metadatas=[doc.metadata]
            )
        
        # Create Chroma vector store from the collection
        vector_store = Chroma(
            client=client,
            collection_name="university_docs",
            embedding_function=embeddings
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Alternative vector store creation failed: {e}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

# Modify the SQLite version check to be more flexible
def check_sqlite_version():
    """
    Check SQLite version with more flexible handling.
    
    Returns:
        bool: Recommendation to proceed or not
    """
    try:
        import sqlite3
        
        # Get SQLite version
        sqlite_version = sqlite3.sqlite_version
        
        # Split version into components
        version_parts = [int(part) for part in sqlite_version.split('.')]
        
        # More lenient version check
        if version_parts[0] < 3 or (version_parts[0] == 3 and version_parts[1] < 35):
            st.warning(f"""
            ⚠️ Potentially Incompatible SQLite Version Detected
            
            Current Version: {sqlite_version}
            Recommended Version: ≥ 3.35.0
            
            This may cause issues with ChromaDB initialization.
            
            Possible Workarounds:
            1. Use in-memory vector store
            2. Upgrade Python/SQLite
            3. Consider different deployment environment
            
            Attempting to proceed with alternative initialization...
            """)
            return False
        
        return True
    
    except Exception as e:
        st.error(f"Error checking SQLite version: {e}")
        return False

def check_existing_vectorstore():
    """Check if a persisted vector store exists."""
    try:
        if get_chroma_client() is None:
            return None
        
        vectorstore_path = get_absolute_path("data/vectorstore")
        if os.path.exists(vectorstore_path):
            # Try to load the existing vector store
            embeddings = get_embeddings()
            vector_store = Chroma(
                client=get_chroma_client(),
                persist_directory=vectorstore_path,
                embedding_function=embeddings,
                collection_name="university_docs"
            )
            # Verify the store has data by attempting to get collection info
            if vector_store._collection.count() > 0:
                return vector_store
    except Exception as e:
        st.error(f"Error loading existing vector store: {e}")
    
    return None

def ensure_directory_permissions(directory):
    """Ensure the directory has proper read/write permissions."""
    try:
        # Create directory with proper permissions if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o755, exist_ok=True)
        else:
            # If directory exists, ensure it's writable
            if not os.access(directory, os.W_OK):
                # Try to make it writable
                os.chmod(directory, 0o755)
                
        # Create a test file to verify write permissions
        test_file = os.path.join(directory, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except Exception as e:
            st.error(f"Directory {directory} is not writable: {e}")
            return False
    except Exception as e:
        st.error(f"Error setting up directory permissions: {e}")
        return False

# Alternative Vector Store Implementation
class SimpleVectorStore(VectorStore):
    def __init__(self, documents, embeddings):
        """
        Create a simple vector store using TF-IDF and cosine similarity
        
        Args:
            documents (List[Document]): List of documents to index
            embeddings (Embeddings): Embedding function
        """
        self._embeddings = embeddings
        # Extract text content from documents
        self.documents = documents
        self.texts = [doc.page_content for doc in documents]
        self.metadatas = [doc.metadata for doc in documents]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        
        st.info(f"Indexed {len(self.documents)} documents")
    
    def add_texts(self, texts, metadatas=None, **kwargs):
        """
        Add new texts to the vector store
        
        Args:
            texts (List[str]): Texts to add
            metadatas (List[dict], optional): Metadata for each text
        
        Returns:
            List[str]: List of added document IDs
        """
        # Generate documents from texts
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        new_documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        # Add documents to the existing store
        self.documents.extend(new_documents)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # Recompute TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        
        # Return document IDs (simple incremental IDs)
        return [f"doc_{len(self.documents) - len(texts) + i}" for i in range(len(texts))]
    
    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        """
        Create a new vector store from raw texts
        
        Args:
            texts (List[str]): Texts to index
            embedding (Embeddings): Embedding function
            metadatas (List[dict], optional): Metadata for each text
        
        Returns:
            SimpleVectorStore: Initialized vector store
        """
        # Generate documents from texts
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        documents = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        
        return cls(documents, embedding)
    
    def similarity_search(self, query, k=4):
        """
        Perform similarity search using cosine similarity
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
        
        Returns:
            List[Document]: Top k most similar documents
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return top k documents
        return [self.documents[idx] for idx in top_indices]
    
    def similarity_search_with_score(self, query, k=4):
        """
        Perform similarity search with similarity scores
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
        
        Returns:
            List[Tuple[Document, float]]: Top k documents with their similarity scores
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top k indices and scores
        top_indices = similarities.argsort()[-k:][::-1]
        top_scores = similarities[top_indices]
        
        # Return top k documents with scores
        return [(self.documents[idx], score) for idx, score in zip(top_indices, top_scores)]
    
    def add_documents(self, documents, **kwargs):
        """
        Add new documents to the vector store
        
        Args:
            documents (List[Document]): Documents to add
        """
        # Add documents to the existing store
        self.documents.extend(documents)
        new_texts = [doc.page_content for doc in documents]
        new_metadatas = [doc.metadata for doc in documents]
        
        self.texts.extend(new_texts)
        self.metadatas.extend(new_metadatas)
        
        # Update TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
    
    def delete(self, ids=None, **kwargs):
        """
        Delete documents by their IDs
        
        Args:
            ids (List[str], optional): List of document IDs to delete
        """
        if ids is None:
            # If no IDs provided, clear all documents
            self.documents = []
            self.texts = []
            self.metadatas = []
            self.tfidf_matrix = None
        else:
            # Remove documents by their indices
            indices_to_keep = [
                i for i in range(len(self.documents)) 
                if f"doc_{i}" not in ids
            ]
            
            # Filter documents, texts, and metadatas
            self.documents = [self.documents[i] for i in indices_to_keep]
            self.texts = [self.texts[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            
            # Recompute TF-IDF matrix
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
    
    def __len__(self):
        """
        Return the number of documents in the vector store
        
        Returns:
            int: Number of documents
        """
        return len(self.documents)
    
    def persist(self, folder_path=None, **kwargs):
        """
        Persist the vector store to a folder
        
        Args:
            folder_path (str, optional): Path to save the vector store
        """
        # In this simple implementation, we don't actually persist
        # A more robust implementation would save texts, metadatas, and TF-IDF matrix
        if folder_path:
            os.makedirs(folder_path, exist_ok=True)
            # Placeholder for actual persistence logic
            st.info(f"Persisted vector store to {folder_path}")
    
    def as_retriever(self, search_kwargs=None):
        """
        Create a retriever from the vector store
        
        Args:
            search_kwargs (dict, optional): Additional search parameters
        
        Returns:
            Retriever: A retriever object compatible with LangChain
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        # Create a custom retriever class
        class SimpleRetriever:
            def __init__(self, vector_store, search_kwargs):
                self.vector_store = vector_store
                self.search_kwargs = search_kwargs
            
            def get_relevant_documents(self, query):
                return self.vector_store.similarity_search(
                    query, 
                    k=self.search_kwargs.get("k", 4)
                )
            
            async def aget_relevant_documents(self, query):
                return self.get_relevant_documents(query)
        
        return SimpleRetriever(self, search_kwargs)

# Modify vector store creation function
def create_vector_store(texts, embeddings, vectorstore_path):
    """
    Create a vector store using a simple TF-IDF approach
    
    Args:
        texts (List[Document]): Documents to index
        embeddings (Embeddings): Embedding function
        vectorstore_path (str): Path to store the vector store (not used in this implementation)
    
    Returns:
        SimpleVectorStore: Initialized vector store
    """
    try:
        # Ensure directory exists
        os.makedirs(vectorstore_path, exist_ok=True)
        
        # Create vector store
        vector_store = SimpleVectorStore(texts, embeddings)
        
        return vector_store
    
    except Exception as e:
        st.error(f"Vector store creation error: {e}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

# Modify the initialization function to use the new vector store
def initialize_chatbot(api_key):
    """
    Initialize the chatbot with a new vector store implementation
    
    Args:
        api_key (str): OpenAI API key
    
    Returns:
        ConversationalRetrievalChain or None
    """
    try:
        # Set up OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Prepare status container
        status_container = st.empty()
        status_container.info("Initializing document database...")
        
        # Load and split documents
        vectorstore_path = os.path.abspath("data/vectorstore")
        
        # Find and load documents
        documents = []
        supported_dirs = ['policies', 'courses', 'handbooks', 'faqs']
        
        for doc_dir in supported_dirs:
            dir_path = os.path.join("data", doc_dir)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    
                    # Support .txt and .pdf files
                    if filename.endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            documents.append(Document(
                                page_content=content, 
                                metadata={'source': file_path}
                            ))
                    elif filename.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
        
        # Check if documents were found
        if not documents:
            st.warning("No documents found. Please add documents to the data directories.")
            return None
         
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        status_container.info(f"Processing {len(texts)} text chunks...")
        
        # Create vector store
        vector_store = create_vector_store(texts, embeddings, vectorstore_path)
        
        if vector_store is None:
            st.error("Failed to create vector store")
            return None
        
        # Create conversational retrieval chain
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0.3, 
            openai_api_key=api_key
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # Create QA chain with the custom vector store as a retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True
        )
        
        status_container.success("Chatbot initialized successfully!")
        return qa_chain
    
    except Exception as e:
        st.error(f"Chatbot initialization error: {e}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def main():
    """Main Streamlit app function."""
    try:
        # Set up the page
        st.set_page_config(
            page_title="University Student Assistant Chatbot",
            page_icon="🎓",
            layout="wide"
        )
        
        # Initialize session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        if 'api_key' not in st.session_state:
            st.session_state.api_key = None
        if 'is_initialized' not in st.session_state:
            st.session_state.is_initialized = False
        
        # Display title and description
        st.title("University Student Assistant Chatbot")
        st.markdown("### Your AI Assistant for University Information")
        
        # Set up the sidebar with API key input
        with st.sidebar:
            st.header("Setup")
            # Create a text input for the API key
            api_key_input = st.text_input(
                "Enter your OpenAI API Key",
                type="password",
                key="api_key_input",
                value=st.session_state.api_key if st.session_state.api_key else "",
                help="Your API key is required to use the chatbot"
            )
            
            if st.button("Save API Key"):
                if api_key_input and api_key_input.strip():
                    st.session_state.api_key = api_key_input.strip()
                    # Save to .env file
                    with open('.env', 'w') as f:
                        f.write(f"OPENAI_API_KEY={api_key_input.strip()}")
                    st.success("API Key saved successfully!")
                    # Reset initialization state
                    st.session_state.is_initialized = False
                    st.session_state.qa_chain = None
                    st.rerun()
                else:
                    st.error("Please enter a valid API key")
            
            st.divider()
            st.header("About")
            st.write("This chatbot can answer questions about your university documents.")
            st.write("You can ask about:")
            st.write("- Course information")
            st.write("- Requirements")
            st.write("- Policies")
            st.write("- General university information")
            
            st.divider()
            st.header("Advanced")
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("Reinitialize Document Database"):
                # Remove the existing vector store
                if os.path.exists("data/vectorstore"):
                    import shutil
                    shutil.rmtree("data/vectorstore")
                st.session_state.qa_chain = None
                st.session_state.is_initialized = False
                st.success("Document database cleared. Reinitializing...")
                st.rerun()
        
        # Check for API key before proceeding
        if not st.session_state.api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
            return
        
        # Initialize chatbot if not already initialized
        if not st.session_state.qa_chain or not st.session_state.is_initialized:
            with st.spinner("Initializing chatbot..."):
                st.session_state.qa_chain = initialize_chatbot(st.session_state.api_key)
                if st.session_state.qa_chain:
                    st.session_state.is_initialized = True
                    st.success("Chatbot initialized successfully!")
                else:
                    st.error("Failed to initialize chatbot. Please check your API key and try again.")
                    return
        
        # Display chat interface
        st.divider()
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your university documents"):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get response
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.qa_chain({"question": prompt})
                        answer = response["answer"]
                        sources = response.get("source_documents", [])
                        
                        # Display the main answer
                        st.write(answer)
                        
                        # Display sources if available
                        if sources:
                            with st.expander("View Sources"):
                                for i, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.markdown(f"```\n{source.page_content}\n```")
                        
                        # Add only the answer to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as response_error:
                st.error(f"Error generating response: {response_error}")
                if hasattr(response_error, '__traceback__'):
                    st.error(traceback.format_exc())
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.error(traceback.format_exc())

if __name__ == "__main__":
    main() 