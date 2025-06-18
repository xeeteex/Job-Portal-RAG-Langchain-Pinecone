# @ IMPORT THE NECESSARY LIBRARIES
import uuid
from langchain_mistralai import ChatMistralAI
from langchain_pinecone import Pinecone as LangChainPinecone
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langsmith import traceable
from src.config.settings import (
    MISTRAL_API_KEY,
    EMBEDDING_MODEL,
    TEMPERATURE,
    MODEL_NAME,
   
)
from src.storage.pinecone_utils import init_pinecone, get_or_create_index

# Helper to wrap Pinecone Index with LangChain Pinecone vectorstore
def get_langchain_pinecone_vectorstore(embeddings, index_name, environment, recreate=True):
    """
    Get or create a Pinecone vector store with the specified parameters.
    
    Args:
        embeddings: The embedding model to use
        index_name: Name of the index
        environment: Cloud environment/region
        recreate: If True, will recreate the index if dimensions don't match
    """
    index = get_or_create_index(embeddings, index_name, environment, recreate=recreate)
    return LangChainPinecone(index, embedding=embeddings, text_key="page_content")

# @ INITIATE THE MISTRAL CLIENT
client = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MODEL_NAME, temperature=TEMPERATURE)


class RagEngine:
    def __init__(self, recreate_index: bool = True):
        """Initialize the RAG engine.
        
        Args:
            recreate_index: If True, will recreate the Pinecone index if dimensions don't match
        """
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info("Initializing embeddings...")
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        
        logger.info("Initializing Pinecone...")
        from src.config.settings import PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT
        
        try:
            self.vectorstore = get_langchain_pinecone_vectorstore(
                self.embeddings, 
                index_name=PINECONE_INDEX_NAME, 
                environment=PINECONE_ENVIRONMENT,
                recreate=recreate_index
            )
            logger.info("Pinecone vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

        print(f"Using Mistral API Key: {MISTRAL_API_KEY[:5]}...")
        self.llm = ChatMistralAI(
            temperature=TEMPERATURE,
            model=MODEL_NAME,
            mistral_api_key=MISTRAL_API_KEY,
            streaming=True,
        )
        template = """You are a Health Care Insurance Data Intrepretor bot. Use the following pieces of context to interpret the user's query. If the information can not be found in the context, just say "I don't know.
        Context: {context}
        Question: {question}
        Answer: """
        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 7}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        print(f"Debug: QA Chain input keys: {self.qa_chain.input_keys}")

    def process_documents(self, docs):
        """Process and store new documents from the vector stores."""
        try:
            self.vectorstore.add_documents(docs)
            print(f"Successfully added {len(docs)} documents to the vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")

    def clear_vectorstore(self):
        """Clear all documents from the vector store"""
        self.vectorstore.delete(delete_all=True)
        print("Vector store cleared.")

    @traceable(run_type="retriever")
    def retriever(self, query: str):
        """Retrieve relevant documents from the vector store"""
        return self.vectorstore.similarity_search(query, k=7)

    @traceable(metadata={"llm": "gpt-3.5-turbo"})
    def interpret_query(self, question, user_id=None):
        run_id = str(uuid.uuid4())
        print(f"Interpreting query: {question}")
        docs = self.retriever(question)
        print(f"Debug: Retrieved {len(docs)} documents")
        print(f"Debug: QA Chain input keys: {self.qa_chain.input_keys}")
        result = self.qa_chain.invoke(
            {"query": question, "input_documents": docs},
            config={"metadata": {"user_id": user_id} if user_id else {}},
        )
        answer = result["result"] if "result" in result else result["answer"]
        sources = [doc.page_content for doc in result["source_documents"]]
        return answer, sources, run_id

    def run_interactive_session(self):
        print("Start talking with the bot (type 'menu' to return to main menu)")

        while True:
            try:
                question = input("\nUser: ").strip()
                if not question:
                    continue
                    
                if question.lower() == "menu":
                    return "menu"

                user_id = input("Enter your user ID (or press enter to skip): ").strip()
                
                try:
                    # Check if we have any documents in the vector store
                    try:
                        test_docs = self.retriever("test")
                        if not test_docs:
                            print("\nℹ️ Note: The vector store appears to be empty. "
                                 "You can still ask questions, but results may be limited. "
                                 "Use the main menu to ingest documents.")
                    except Exception as e:
                        print(f"\n⚠️ Warning: Could not access vector store: {str(e)}")
                    
                    # Process the query
                    answer, sources, run_id = self.interpret_query(
                        question, user_id if user_id else None
                    )
                    
                    print(f"\n🤖 Answer: {answer}")
                    if sources and len(sources) > 0:
                        print(f"\n📚 Sources found: {len(sources)}")
                    print(f"🔗 Run ID: {run_id}")

                    # Feedback and continue handling
                    if answer.strip().lower() not in ["i don't know.", "i don't know", ""]:
                        # Get feedback
                        while True:
                            feedback = input("\nWas this answer helpful? (y/n): ").strip().lower()
                            if feedback in ['y', 'n']:
                                self.log_feedback(run_id, 1 if feedback == 'y' else 0)
                                break
                            print("Please enter 'y' for yes or 'n' for no.")
                        
                        # Ask to continue
                        while True:
                            cont = input("\nWould you like to ask another question? (y/n): ").strip().lower()
                            if cont == 'n':
                                return "menu"
                            elif cont == 'y':
                                break
                            print("Please enter 'y' to continue or 'n' to return to main menu.")
                
                except Exception as e:
                    print(f"\n❌ Error processing your query: {str(e)}")
                    
                    # Ask if user wants to continue after an error
                    while True:
                        choice = input("\nContinue? (y/n/menu): ").strip().lower()
                        if choice == 'n':
                            return "exit"
                        elif choice == 'menu':
                            return "menu"
                        elif choice == 'y':
                            break
                        print("Please enter 'y' to continue, 'n' to exit, or 'menu' to return to main menu.")
            
            except KeyboardInterrupt:
                print("\n\n🛑 Operation cancelled.")
                return "menu"

            print("\n")

    def log_feedback(self, run_id, score):
        from langsmith import Client

        ls_client = Client()
        ls_client.create_feedback(run_id=run_id, score=score, key="user_score")

    def get_qa_chain(self):
        return self.qa_chain
