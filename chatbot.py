import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def get_api_key():
    """Get OpenAI API key from environment or user input."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        print("\nWelcome to the University Student Assistant Chatbot!")
        print("To get started, you need an OpenAI API key.")
        print("You can get one from: https://platform.openai.com/api-keys")
        print("\nPlease enter your OpenAI API key:")
        api_key = input().strip()
        
        # Save the API key to .env file
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}")
            
        print("\nAPI key saved! You won't need to enter it again.")
        print("-" * 50)
    
    return api_key

class UniversityChatbot:
    def __init__(self):
        self.api_key = get_api_key()
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def initialize_knowledge_base(self, data_dir="data"):
        """Initialize the knowledge base with university documents."""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"\nCreated {data_dir} directory. Please add your university documents there.")
            return False
            
        # Load documents from the data directory
        loader = DirectoryLoader(data_dir, glob="**/*.txt")
        documents = loader.load()
        
        if not documents:
            print("\nNo documents found in the data directory.")
            print("Please add your university documents to the 'data' folder.")
            return False
            
        print("\nLoading documents...")
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="chroma_db"
        )
        
        # Initialize the QA chain
        llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )
        
        print("Documents loaded successfully!")
        return True
        
    def get_response(self, query):
        """Get response from the chatbot."""
        if not self.qa_chain:
            return "Please initialize the knowledge base first by adding documents to the data directory."
            
        try:
            result = self.qa_chain({"question": query})
            return result["answer"]
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Initialize the chatbot
    chatbot = UniversityChatbot()
    
    # Initialize knowledge base
    if not chatbot.initialize_knowledge_base():
        print("\nTo add documents:")
        print("1. Create a 'data' folder if it doesn't exist")
        print("2. Add your university documents (text files) to the 'data' folder")
        print("3. Run the chatbot again")
        return
        
    print("\nUniversity Student Assistant Chatbot")
    print("Type 'quit' to exit")
    print("Type 'help' to see available commands")
    print("-" * 50)
    
    while True:
        query = input("\nYour question: ").strip()
        
        if query.lower() == 'quit':
            print("\nThank you for using the University Student Assistant!")
            break
        elif query.lower() == 'help':
            print("\nAvailable commands:")
            print("- 'quit': Exit the chatbot")
            print("- 'help': Show this help message")
            print("\nYou can ask questions about:")
            print("- Course information")
            print("- Requirements")
            print("- Schedule")
            print("- Grading")
            print("- Contact information")
            continue
            
        response = chatbot.get_response(query)
        print("\nAssistant:", response)

if __name__ == "__main__":
    main() 