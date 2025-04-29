import os
import pytest
from fastapi.testclient import TestClient
from main import app

# Initialize the FastAPI test client
client = TestClient(app)

# Sample Python code to be used for upload testing
SAMPLE_CODE = """
def greet(name):
    \"\"\"Greets a person by name.\"\"\"
    return f"Hello, {name}"

class Greeter:
    def __init__(self, prefix="Hi"):
        self.prefix = prefix

    def greet(self, name):
        return f"{self.prefix}, {name}"
"""

# Path where the sample code will be temporarily saved for tests
TEST_FILE_PATH = "tests/sample_code.py"

# Fixture to set up and tear down the sample code file used in tests
@pytest.fixture(scope="module", autouse=True)
def create_sample_code_file():
    """
    Creates a temporary Python file before tests run and deletes it afterward.
    Ensures all tests have consistent access to the same file input.
    """
    os.makedirs("tests", exist_ok=True)
    with open(TEST_FILE_PATH, "w") as f:
        f.write(SAMPLE_CODE)
    yield
    os.remove(TEST_FILE_PATH)

# Health check endpoint test
def test_health_check():
    """
    Tests the root endpoint to verify the API is up and running.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Code Analyzer API is running with Azure OpenAI and agent architecture."

# Successful file upload test
def test_file_upload_success():
    """
    Tests uploading a valid Python file and checks for expected analysis keys in the response.
    """
    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post("/upload/", files={"file": ("sample_code.py", f, "text/x-python")})
    assert response.status_code == 200
    data = response.json()
    # Check for all analysis components in the response
    assert "summary" in data
    assert "docstring" in data
    assert "inline_comments" in data
    assert "pylint_analysis" in data
    assert "bandit_analysis" in data

# Invalid file format upload test
def test_file_upload_invalid_format():
    """
    Tests uploading an unsupported file format and expects a 400 error.
    """
    response = client.post("/upload/", files={"file": ("bad.txt", b"print('oops')", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type."

# Query processing test
def test_query_after_upload():
    """
    Tests querying the system after file upload to check AI understanding and response.
    """
    response = client.post("/query", json={"query": "What does the greet function do?"})
    assert response.status_code == 200
    assert "response" in response.json()

# Confirmation endpoint test
def test_confirm_response():
    """
    Tests submitting user feedback to confirm usefulness of the response.
    """
    response = client.post("/confirm", json={"feedback": "That was helpful!"})
    assert response.status_code == 200
    assert response.json()["message"] == "Thanks for the feedback!"
