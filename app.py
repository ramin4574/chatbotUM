import streamlit as st
import os
import sys
import traceback
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document

# Explicitly handle ChromaDB import with comprehensive error handling
try:
    # Attempt to modify system path if needed
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Try importing with different strategies
    try:
        import chromadb
    except ImportError:
        # If standard import fails, try alternative import
        import importlib
        chromadb = importlib.import_module('chromadb')
    
    # Explicitly import Settings if needed
    from chromadb.config import Settings
except Exception as e:
    st.error(f"Critical ChromaDB Import Error: {e}")
    st.error(f"Import Traceback: {traceback.format_exc()}")
    raise RuntimeError(f"Cannot import ChromaDB: {e}")

def get_api_key():
    """
    Retrieve the OpenAI API key from environment or user input.
    Ensures a visible and clear method to input the API key.
    """
    # First, try to load from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If no API key is found, create a prominent input method
    if not api_key:
        # Create an input field for the API key
        api_key = st.text_input(
            "Enter your OpenAI API Key", 
            type="password", 
            key="openai_api_key_input",
            help="Your API key is used to authenticate with OpenAI's services"
        )
        
        # Add a button to save the API key
        if st.button("Save API Key", key="save_api_key_button"):
            if api_key and api_key.strip():
                # Save to .env file
                with open('.env', 'w') as f:
                    f.write(f"OPENAI_API_KEY={api_key.strip()}")
                st.success("API Key saved successfully!")
                # Explicitly set the environment variable
                os.environ["OPENAI_API_KEY"] = api_key.strip()
                # Rerun the app to use the new key
                st.experimental_rerun()
            else:
                st.error("Please enter a valid API key")
        
        # If no key is entered, return None
        if not api_key or not api_key.strip():
            return None
    
    return api_key

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

# Ensure data directory exists with explicit permissions
os.makedirs("data/vectorstore", exist_ok=True)
os.chmod("data/vectorstore", 0o755)

# Robust ChromaDB client initialization
def get_chroma_client():
    try:
        # Use absolute path and explicit error handling
        vectorstore_path = os.path.abspath("data/vectorstore")
        
        # Ensure directory exists and is writable
        os.makedirs(vectorstore_path, exist_ok=True)
        os.chmod(vectorstore_path, 0o755)
        
        # Initialize client with comprehensive error handling
        client = chromadb.PersistentClient(path=vectorstore_path)
        
        # Verify client initialization
        if client is None:
            raise ValueError("Failed to initialize ChromaDB client")
        
        return client
    except Exception as e:
        st.error(f"ChromaDB Client Initialization Error: {e}")
        st.error(f"Error Details: {traceback.format_exc()}")
        return None

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

def initialize_chatbot(api_key):
    """Initialize the chatbot with documents."""
    try:
        st.write("Starting initialization...")
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Check for existing vector store first
        vector_store = check_existing_vectorstore()
        if vector_store:
            st.success("📚 Loading existing document database...")
        else:
            st.info("🔄 Creating new document database...")
            # Load documents from the data directory
            documents = []
            
            # Directories to load
            base_data_dir = get_absolute_path("data")
            directories_to_load = [
                os.path.join(base_data_dir, "policies"),
                os.path.join(base_data_dir, "courses"),
                os.path.join(base_data_dir, "handbooks"),
                os.path.join(base_data_dir, "faqs")
            ]
            
            # Create directories if they don't exist
            for directory in directories_to_load:
                os.makedirs(directory, exist_ok=True)
            
            # Create vectorstore directory
            vectorstore_path = get_absolute_path("data/vectorstore")
            os.makedirs(vectorstore_path, exist_ok=True)
            
            st.info("""
            📚 Document Locations:
            - Place policy documents in: `data/policies/`
            - Place course materials in: `data/courses/`
            - Place handbooks in: `data/handbooks/`
            - Place FAQs in: `data/faqs/`
            
            Supported formats: PDF and TXT files
            
            To update the knowledge base:
            1. Add new documents to these folders
            2. Click "Reinitialize Document Database" in the sidebar
            """)
            
            # Create a status container
            status_container = st.empty()
            progress_container = st.empty()
            my_bar = progress_container.progress(0)
            
            # First scan for files
            total_files = 0
            total_size = 0
            file_list = []
            
            # Scan all directories first
            for directory in directories_to_load:
                try:
                    if os.path.exists(directory):
                        files = [f for f in os.listdir(directory) if not f.startswith('.') and (f.endswith('.pdf') or f.endswith('.txt'))]
                        for f in files:
                            file_path = os.path.join(directory, f)
                            try:
                                if os.path.isfile(file_path):
                                    size = os.path.getsize(file_path)
                                    file_list.append((directory, f, size))
                                    total_size += size
                                    total_files += 1
                            except OSError as e:
                                st.error(f"Error getting size for {file_path}: {e}")
                except Exception as e:
                    st.error(f"Error accessing directory {directory}: {e}")
            
            if total_files == 0:
                st.error("No valid documents found in the specified directories.")
                return None
            
            status_container.info(f"Found {total_files} files (Total size: {get_file_size(total_size)})")
            
            # Process files
            files_processed = 0
            size_processed = 0
            
            for directory, file, size in file_list:
                try:
                    file_path = os.path.join(directory, file)
                    if not os.path.isfile(file_path):
                        st.error(f"File not found: {file_path}")
                        continue
                        
                    status_container.info(f"Processing {file} ({get_file_size(size)})...")
                    
                    try:
                        if file.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                            pdf_docs = loader.load()
                            # Add debug logging
                            st.info(f"Extracted {len(pdf_docs)} pages from {file}")
                            for i, doc in enumerate(pdf_docs):
                                st.text(f"Page {i+1} preview: {doc.page_content[:200]}...")
                            documents.extend(pdf_docs)
                        elif file.endswith(".txt"):
                            with open(file_path, 'r') as f:
                                content = f.read()
                                documents.append(Document(page_content=content, metadata={"source": file_path}))
                        
                        files_processed += 1
                        size_processed += size
                        progress = size_processed / total_size if total_size > 0 else 0
                        my_bar.progress(progress, f"Processing files... ({files_processed}/{total_files})")
                        
                    except Exception as file_error:
                        st.error(f"Error loading file {file_path}: {file_error}")
                        
                except Exception as list_error:
                    st.error(f"Error processing file {file}: {list_error}")
                    
            if not documents:
                st.error("No documents could be loaded successfully.")
                return None
            
            status_container.info("Splitting documents into chunks...")
            
            # Split documents into chunks
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,  # Increased chunk size
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                status_container.info(f"Split into {len(texts)} chunks")
                # Add debug logging for chunks
                st.info("Preview of first few chunks:")
                for i, chunk in enumerate(texts[:3]):
                    st.text(f"Chunk {i+1} preview: {chunk.page_content[:200]}...")
                
                if not texts:
                    st.error("No text chunks were created from the documents.")
                    return None
                    
            except Exception as split_error:
                st.error(f"Error splitting documents: {split_error}")
                return None
            
            # Create vector store
            try:
                status_container.info("Creating vector store (this may take a few minutes)...")
                my_bar.progress(1.0)
                
                # Ensure directory exists and is writable
                vectorstore_path = get_absolute_path("data/vectorstore")
                os.makedirs(vectorstore_path, mode=0o755, exist_ok=True)
                
                embeddings = get_embeddings()
                
                # Create vector store with explicit client
                vector_store = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    client=get_chroma_client(),
                    persist_directory=vectorstore_path,
                    collection_name="university_docs"
                )
                vector_store.persist()
                status_container.success("Vector store created and persisted successfully!")
                
            except Exception as vector_store_error:
                error_msg = str(vector_store_error)
                st.error(f"Error creating vector store: {vector_store_error}")
                return None
            
            # Clean up progress indicators
            progress_container.empty()
            
        # Initialize the QA chain
        try:
            status_container.info("Initializing QA chain...")
            
            # Create a custom prompt template for more detailed answers
            template = """You are a helpful university assistant with access to university documents. 
            When answering questions, please:
            1. Provide comprehensive, detailed answers
            2. Include all relevant information from the documents
            3. Use bullet points or numbered lists when appropriate
            4. Give examples when possible
            5. Quote specific sections when relevant
            
            Context: {context}
            
            Question: {question}
            
            Chat History: {chat_history}
            
            Please provide a detailed answer:"""
            
            from langchain.prompts import PromptTemplate
            QA_PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question", "chat_history"]
            )
            
            llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo-16k"  # Using 16k model for more context
            )
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(search_kwargs={
                    "k": 8  # Increased number of chunks for more comprehensive context
                }),
                memory=memory,
                return_source_documents=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={"prompt": QA_PROMPT}
            )
            
            status_container.success("Ready to answer questions!")
            return qa_chain
            
        except Exception as qa_chain_error:
            st.error(f"Error initializing QA chain: {qa_chain_error}")
            return None
            
    except Exception as unexpected_error:
        st.error(f"Unexpected error in chatbot initialization: {unexpected_error}")
        st.error(traceback.format_exc())
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
                    st.experimental_rerun()
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