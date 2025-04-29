import os
import io
import logging
import ast
import re
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from azure.identity import DefaultAzureCredential
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from radon.complexity import cc_visit
from azure.ai.openai import OpenAIClient, ChatCompletionRequestMessage, ChatCompletionsOptions

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'py', 'js', 'java'}

# Initialize embedding model and FAISS index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
embedding_to_code_map = {}

# Path to save/load FAISS index
FAISS_INDEX_PATH = "faiss_index.index"

# Function to save the FAISS index to disk
def save_faiss_index():
    faiss.write_index(index, FAISS_INDEX_PATH)

# Function to load the FAISS index from disk
def load_faiss_index():
    global index
    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        # If no index exists, create a new one
        index = faiss.IndexFlatL2(dimension)

# Load FAISS index on startup
load_faiss_index()

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_OPENAI_DEPLOYMENT]):
    raise ValueError("Azure OpenAI environment variables are not set.")

openai_client = OpenAIClient(
    endpoint=AZURE_OPENAI_ENDPOINT,
    credential=DefaultAzureCredential()
)

# In-memory storage for uploaded code
uploaded_code = {}

# In-memory pending responses store for human-in-the-loop
pending_responses = {}

# Pydantic model for query payload
class QueryPayload(BaseModel):
    query: str

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to generate docstring for Python code
def generate_docstring(code):
    tree = ast.parse(code)
    docstrings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = f"Function '{node.name}'\n"
            args = [arg.arg for arg in node.args.args]
            docstring += f"Arguments: {', '.join(args)}\n"
            if node.args.args:
                arg_types = []
                for arg in node.args.args:
                    if arg.annotation:
                        arg_types.append(f"{arg.arg}: {arg.annotation}")
                if arg_types:
                    docstring += f"Argument types: {', '.join(arg_types)}\n"
            if node.returns:
                docstring += f"Return type: {node.returns}\n"
            docstring += f"Docstring: {ast.get_docstring(node) or 'No docstring provided.'}\n"
            docstrings.append(docstring)
        elif isinstance(node, ast.ClassDef):
            docstring = f"Class '{node.name}'\n"
            docstring += f"Docstring: {ast.get_docstring(node) or 'No docstring provided.'}\n"
            docstrings.append(docstring)
    return '\n'.join(docstrings)

# Function to generate a high-level summary of the code
def generate_code_summary(code):
    lines = code.split("\n")
    summary = []

    num_classes = len([line for line in lines if line.strip().startswith("class ")])
    num_functions = len([line for line in lines if line.strip().startswith("def ")])
    summary.append(f"Number of classes: {num_classes}")
    summary.append(f"Number of functions: {num_functions}")
    summary.append(f"Total lines of code: {len(lines)}")
    summary.append("Code contains functions and classes for core logic.")
    return "\n".join(summary)

# Function to add inline comments to the code
def add_inline_comments(code):
    lines = code.split("\n")
    commented_code = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("def "):
            commented_code.append(f"{line}  # Function definition")
        elif stripped_line.startswith("class "):
            commented_code.append(f"{line}  # Class definition")
        elif "return" in stripped_line:
            commented_code.append(f"{line}  # Return statement")
        elif "for " in stripped_line or "while " in stripped_line:
            commented_code.append(f"{line}  # Loop")
        else:
            commented_code.append(line)
    return "\n".join(commented_code)


def detect_bias_or_sensitive_content(text):
    bias_keywords = ["gender", "race", "religion", "ethnicity", "violence", "political"]
    return any(kw in text.lower() for kw in bias_keywords)


    # Function to analyze code with Pylint
def analyze_code_with_pylint(filepath):
    try:
        result = subprocess.run(
            ['pylint', '--output-format=text', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        pylint_output = result.stdout
        pylint_error = result.stderr
        
        if result.returncode != 0:
            logging.error(f"Pylint execution failed with exit code {result.returncode}")
            return f"Error during pylint execution: {pylint_error}"

        return pylint_output
    
    except Exception as e:
        logging.error(f"Error during pylint execution: {str(e)}")
        return f"Error during pylint execution: {str(e)}"


# Function to analyze code complexity with Radon
def analyze_code_with_radon(filepath):
    with open(filepath, 'r') as file:
        code = file.read()
    complexity = cc_visit(code)
    return complexity

# Split code into chunks
def split_code_into_chunks(code, max_lines=20):
    lines = code.split('\n')
    chunks = []
    for i in range(0, len(lines), max_lines):
        chunk = "\n".join(lines[i:i + max_lines])
        chunks.append(chunk)
    return chunks

# Function to detect common code smells
def detect_code_smells(code):
    smells = []
    functions = re.findall(r"def .+?:\n(?:\s{4}.+\n)+", code)
    for func in functions:
        num_lines = len(func.split("\n"))
        if num_lines > 20:
            smells.append("Code smell: Function exceeds 20 lines.")
    if code.count(code[:100]) > 1:
        smells.append("Code smell: Possible duplicate code detected.")
    return smells

# Function to detect potential performance issues
def detect_performance_issues(code):
    issues = []
    if re.search(r"for .+ in .+:\s*for .+ in .+", code):
        issues.append("Performance issue: Nested loops found. Could be optimized.")
    if re.search(r"for .+ in .+:\s*.+\.append\(.+\)", code):
        issues.append("Performance issue: List append in loop. Consider using a list comprehension.")
    return issues

# Function to generate refactoring suggestions
def generate_refactoring_suggestions(code):
    suggestions = []
    functions = re.findall(r"def .+?:\n(?:\s{4}.+\n)+", code)
    for func in functions:
        num_lines = len(func.split("\n"))
        if num_lines > 30:
            suggestions.append("Refactoring suggestion: Function is too large. Consider splitting it.")
    if "for " in code and ".append" in code:
        suggestions.append("Refactoring suggestion: Replace loop with list comprehension where possible.")
    return suggestions


# Function to analyze code with bandit
def analyze_code_with_bandit(filepath):
    try:
        bandit_output = subprocess.run(
            ['bandit', '-r', filepath, '-f', 'json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return bandit_output.stdout
    except Exception as e:
        logging.error(f"Error during bandit execution: {str(e)}")
        return f"Error during bandit execution: {str(e)}"

# Route to upload and analyze a file
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    uploaded_code[file.filename] = code
    embedding = embedding_model.encode(code)
    embedding = np.array([embedding])
    index.add(embedding)
    embedding_to_code_map[len(embedding_to_code_map)] = code

        # Save FAISS index after adding embeddings
    save_faiss_index()

    docstring = generate_docstring(code)
    summary = generate_code_summary(code)
    inline_comments = add_inline_comments(code)
    code_smells = detect_code_smells(code)
    performance_issues = detect_performance_issues(code)
    refactoring = generate_refactoring_suggestions(code)
    pylint_report = analyze_code_with_pylint(file_path)
    bandit_report = analyze_code_with_bandit(file_path)

    return {
        "filename": file.filename,
        "summary": summary,
        "docstring": docstring,
        "inline_comments": inline_comments,
        "code_smells": code_smells,
        "performance_issues": performance_issues,
        "refactoring_suggestions": refactoring,
        "pylint_analysis": pylint_report,
        "bandit_analysis": bandit_report,
    }


def search_code_snippets(query, top_k=1):
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx in embedding_to_code_map:
            results.append(embedding_to_code_map[idx])
    return results

# Function to handle queries related to the code
@app.post("/query")
async def query_code(payload: QueryPayload):
    if not uploaded_code:
        raise HTTPException(status_code=400, detail="No code uploaded yet.")

    query = payload.query

    # 🔍 Step 1: Retrieve the most relevant code snippet using FAISS
    relevant_snippets = search_code_snippets(query, top_k=1)
    if not relevant_snippets:
        raise HTTPException(status_code=404, detail="No relevant code snippet found.")

    relevant_code = relevant_snippets[0]

    # 🤖 Step 2: Agentic response using ChatCompletions
    messages = [
        ChatCompletionRequestMessage(role="system", content="You are an expert coding agent. When given code, you should first think aloud, summarize the code, find key functions, detect bugs, and only then answer the user's question step-by-step."),
        ChatCompletionRequestMessage(role="user", content=f"Code snippet:\n{relevant_code}"),
        ChatCompletionRequestMessage(role="user", content=f"My question: {query}")
    ]

    response = openai_client.get_chat_completions(
        deployment_id=AZURE_OPENAI_DEPLOYMENT,
        chat_completions_options=ChatCompletionsOptions(
            messages=messages,
            max_tokens=500,
            temperature=0.3,
            n=1
        )
    )

    return {"response": response.choices[0].message.content.strip()}

    # Bias Detection
    if detect_bias_or_sensitive_content(answer):
        logging.warning(f"Bias/Sensitive content detected: {answer}")
        answer = "⚠️ Warning: Sensitive content detected. Review carefully.\n\n" + answer

    # Save for Human-in-the-Loop confirmation
    response_id = str(uuid.uuid4())
    pending_responses[response_id] = answer
    logging.info(f"Pending response stored: {response_id}")

    return {
        "response_id": response_id,
        "response_preview": answer,
        "status": "pending_confirmation"
    }

# Confirm/Reject endpoint
@app.post("/confirm")
async def confirm_response(payload: ConfirmPayload):
    if payload.response_id not in pending_responses:
        raise HTTPException(status_code=404, detail="Response not found.")

    if payload.confirm:
        final_response = pending_responses.pop(payload.response_id)
        logging.info(f"Response confirmed: {payload.response_id}")
        return {
            "status": "confirmed",
            "final_response": final_response
        }
    else:
        pending_responses.pop(payload.response_id)
        logging.info(f"Response rejected: {payload.response_id}")
        return {
            "status": "rejected",
            "message": "Response rejected. Please submit a new query."
        }

        
# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Code Analyzer API is running with Azure OpenAI and agent architecture."}

# Run the application   
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)