# Intelligent RAG Assistant

An AI-powered HR assistant that answers employee questions about company
policies using Retrieval-Augmented Generation (RAG). It retrieves
relevant information from a local collection of HR PDF documents and
uses an LLM to generate grounded responses. An additional scope-checking
step helps keep the assistant focused on HR-related questions.

## Features

-   **PDF knowledge base:** Loads HR policy documents from a local
    folder.
-   **Text chunking:** Splits documents into overlapping chunks for
    retrieval.
-   **Semantic embeddings:** Uses
    `sentence-transformers/all-MiniLM-L6-v2` through Hugging Face
    embeddings.
-   **Vector search:** Stores and retrieves relevant chunks with FAISS.
-   **LLM responses:** Uses Groq through LangChain; the notebook
    configures `openai/gpt-oss-120b` by default.
-   **Scope guard:** Classifies questions as in-scope or out-of-scope
    before answering.
-   **Source references:** Returns retrieved documents alongside the
    generated answer.
-   **Batch evaluation:** Reads questions from `test.csv` and writes
    answers to `submission.csv`.

## Workflow

1.  Load HR PDFs from `./zyro-dynamics-hr-corpus`.
2.  Split document text into chunks (800 characters, with 100-character
    overlap).
3.  Create embeddings for the chunks.
4.  Index the embeddings in a FAISS vector store.
5.  Retrieve the top 3 relevant chunks for each question.
6.  Check whether the question is related to HR.
7.  Generate an answer using the retrieved context, or return a
    scope-limiting message.
8.  Optionally process the evaluation questions in `test.csv` and export
    `submission.csv`.

## Tech Stack

-   Python
-   Jupyter Notebook
-   LangChain
-   Groq API
-   Hugging Face Sentence Transformers
-   FAISS
-   PyPDF
-   Pandas
-   LangSmith tracing (optional)

## Project Structure

``` text
project/
├── code(1).ipynb
├── .env                 # Create locally; do not commit
├── test.csv             # Evaluation questions
├── submission.csv       # Generated answers
└── zyro-dynamics-hr-corpus/
    ├── hr-policy-1.pdf
    └── hr-policy-2.pdf
```

The `zyro-dynamics-hr-corpus` folder should contain the HR PDF files
used as the assistant's knowledge base.

## Setup

### 1. Create and activate a virtual environment (recommended)

``` bash
python -m venv .venv
```

Windows:

``` bash
.venv\Scripts\activate
```

macOS/Linux:

``` bash
source .venv/bin/activate
```

### 2. Install dependencies

Run this in a notebook cell or terminal:

``` bash
pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-groq langchain-google-genai langchain-openai langchain-core faiss-cpu pypdf sentence-transformers python-dotenv langsmith pandas
```

### 3. Configure the API key

Create a `.env` file in the project root:

``` env
GROQ_API_KEY=your_groq_api_key_here
```

Never paste a real API key into source code or commit it to GitHub. Add
`.env` to `.gitignore`. If a real key has been exposed, revoke it and
create a replacement.

### 4. Check the configuration

The notebook uses these defaults:

``` python
LLM_Provider = "groq"
LLM_Model = "openai/gpt-oss-120b"
CORPUS_PATH = "./zyro-dynamics-hr-corpus"
```

Make sure the model identifier is available to your Groq account and the
corpus path points to the folder containing your PDFs.

## Run the Project

1.  Open `code(1).ipynb` in Jupyter Notebook or VS Code.
2.  Run the dependency installation and import cells.
3.  Ensure `.env` is loaded and `GROQ_API_KEY` is available.
4.  Confirm the HR PDF folder exists at the configured path.
5.  Run the notebook cells in order to load documents, build the FAISS
    index, initialize the LLM, and create the RAG and scope-guard
    chains.
6.  Test a question such as:

``` python
result = ask_bot("How do I apply for parental leave?")
print("Answer:", result["answer"])
print("Sources:", [doc.metadata.get("source") for doc in result["sources"]])
```

## Generate Submission Answers

Place `test.csv` in the project root. It should include `question_id`
and `question` columns. Run the notebook's batch-processing cell to
generate `submission.csv` with `question_id` and `answer` columns.

## Configuration Notes

-   The retriever currently returns the top 3 chunks
    (`search_kwargs={"k": 3}`).
-   Chunk size is 800 characters with 100 characters of overlap.
-   The generation temperature is set to `0.5`, with a maximum of 500
    output tokens.
-   LangSmith tracing is enabled by the `@traceable` decorator. If you
    do not need tracing, remove the decorator or install/configure
    LangSmith as required.
-   The notebook imports multiple provider integrations, but the current
    model configuration uses Groq.


