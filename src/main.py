import os
from src.rag.engine import RagEngine
from src.monitoring.langsmith_monitor import LangSmithMonitor
from src.data_processing.document_loader import load_documents
from src.config.settings import DATA_DIR


class HealthCareBot:
    def __init__(self):
        print("Initializing Health Care Bot...")
        self.engine: RagEngine | None = None
        self.monitor: LangSmithMonitor | None = None
        self.documents_ingested: bool = False
        
        # Check required environment variables
        required_vars = [
            'OPENAI_API_KEY',
            'PINECONE_API_KEY',
            'PINECONE_INDEX_NAME',
            'PINECONE_ENVIRONMENT',
            'EMBEDDING_MODEL'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
            print("Please check your .env file and ensure all required variables are set.")
            return
            
        # Initialize components
        try:
            print("Initializing RAG engine...")
            self.engine = RagEngine()
            print("RAG engine initialized successfully.")
            
            print("Initializing LangSmith monitor...")
            self.monitor = LangSmithMonitor()
            print("LangSmith monitor initialized successfully.")
            
        except ImportError as e:
            print(f"Import error: {e}")
            print("Please make sure all required packages are installed.")
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            import traceback
            traceback.print_exc()

    def run_interactive(self):
        print("Starting interactive session...")
        try:
            result = self.engine.run_interactive_session()
            if result == "exit":
                print("Goodbye!")
                return "exit"
            # If we get here, we're either returning to menu or continuing
            # The engine will handle the case when no documents are available
        except Exception as e:
            print(f"An error occurred during the interactive session: {e}")
            import traceback
            traceback.print_exc()
            print("Returning to main menu.")

    def run_monitoring(self, start_time, end_time):
        print(f"Running monitoring for period: {start_time} - {end_time}")
        try:
            report = self.monitor.generate_report(start_time, end_time)
            print(report)
        except Exception as e:
            print(f"An error occurred while generating the report: {e}")

    def ingest_documents(self):
        if self.engine is None:
            print("Error: Engine is not properly initialized. Cannot ingest documents.")
            print("Please check the initialization errors above and fix them.")
            return
            
        print("Ingesting documents...")
        try:
            # Check if data directory exists
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            if not os.path.exists(data_dir) or not os.listdir(data_dir):
                print(f"Error: No documents found in {data_dir}")
                print("Please place your PDF files in the 'data' directory and try again.")
                return
                
            # Load documents as single chunks
            print("Loading documents...")
            docs = load_documents()
            if not docs:
                print("No documents found or no PDF files in the data directory.")
                return
                
            print(f"Processing {len(docs)} document chunks...")
            self.engine.process_documents(docs)
            self.documents_ingested = True
            print(f"Successfully ingested {len(docs)} document chunks.")
            
        except FileNotFoundError as e:
            print(f"Error: Directory not found - {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()

    def clear_data(self):
        confirmation = input(
            "Are you sure you want to clear all data? This cannot be undone. (y/n): "
        )
        if confirmation.lower() == "y":
            self.engine.clear_vectorstore()
            print("All data has been cleared.")
        else:
            print("Operation cancelled.")


def main():
    print("Starting Health Care Bot...")

    bot = HealthCareBot()
    while True:
        print("\nMain Menu:")
        print("1. Ingest documents")
        print("2. Run interactive session")
        print("3. Run monitoring")
        print("4. Clear data")
        print("5. Exit")
        
        choice = input("\nChoose an option (1-5): ").strip()
        
        if choice == "1":
            bot.ingest_documents()
        elif choice == "2":
            result = bot.run_interactive()
            if result == "exit":
                break
        elif choice == "3":
            try:
                print("\nEnter date range for monitoring (YYYY-MM-DD format):")
                start_time = input("Start date: ").strip()
                end_time = input("End date: ").strip()
                bot.run_monitoring(start_time, end_time)
            except Exception as e:
                print(f"Error running monitoring: {str(e)}")
        elif choice == "4":
            bot.clear_data()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")


if __name__ == "__main__":
    main()
