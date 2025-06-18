# Job Portal RAG with LangChain and Pinecone

## Description
A Retrieval-Augmented Generation (RAG) system designed for job recruitment. This application processes resumes and other documents, stores them in a vector database (Pinecone), and provides an interactive interface to query the documents using natural language.

## ✨ Features

- **Document Processing**: Supports PDF, TXT, and DOCX formats
- **Vector Storage**: Uses Pinecone for efficient similarity search
- **Language Models**: Integration with Mistral AI and OpenAI models
- **Interactive CLI**: User-friendly command-line interface
- **Monitoring**: LangSmith integration for tracking and analytics
- **API**: RESTful endpoints for integration with other services

## 🚀 Prerequisites

- Python 3.8+
- [Pinecone](https://www.pinecone.io/) account and API key
- [Mistral AI](https://mistral.ai/) API key
- [LangSmith](https://smith.langchain.com/) API key (optional, for monitoring)
- [OpenAI](https://platform.openai.com/) API key (optional, alternative to Mistral AI)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Job-Portal-RAG-Langchain-Pinecone.git
   cd Job-Portal-RAG-Langchain-Pinecone
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with the following content:
   ```env
   # Required
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=your_index_name
   PINECONE_ENVIRONMENT=your_environment
   MISTRAL_API_KEY=your_mistral_api_key
   
   # Optional
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   TEMPERATURE=0.7
   MODEL_NAME=mistral-tiny
   ```

## 🚀 Usage

### CLI Mode
1. **Prepare your documents**
   - Place your PDF, TXT, or DOCX files in the `data/tmp` directory
   - The application will automatically process and index these documents

2. **Run the application**
   ```bash
   # Windows
   python -m src.main
   
   # macOS/Linux
   python3 -m src.main
   ```

3. **Follow the interactive menu**
   - Ingest documents
   - Run interactive Q&A session
   - View monitoring reports
   - Clear existing data

### API Mode
Start the FastAPI server:
```bash
python -m uvicorn src.api.server:app --reload
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`

## 📊 Monitoring

To enable LangSmith monitoring:
1. Sign up at [LangSmith](https://smith.langchain.com/)
2. Get your API key and add it to the `.env` file
3. The application will automatically log all interactions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
