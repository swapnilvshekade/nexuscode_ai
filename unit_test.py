import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

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

TEST_FILE_PATH = "tests/sample_code.py"


@pytest.fixture(scope="module", autouse=True)
def create_sample_code_file():
    os.makedirs("tests", exist_ok=True)
    with open(TEST_FILE_PATH, "w") as f:
        f.write(SAMPLE_CODE)
    yield
    os.remove(TEST_FILE_PATH)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Code Analyzer API is running with Azure OpenAI and agent architecture."


def test_file_upload_success():
    with open(TEST_FILE_PATH, "rb") as f:
        response = client.post("/upload/", files={"file": ("sample_code.py", f, "text/x-python")})
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "docstring" in data
    assert "inline_comments" in data
    assert "pylint_analysis" in data
    assert "bandit_analysis" in data


def test_file_upload_invalid_format():
    response = client.post("/upload/", files={"file": ("bad.txt", b"print('oops')", "text/plain")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type."


def test_query_after_upload():
    response = client.post("/query", json={"query": "What does the greet function do?"})
    assert response.status_code == 200
    assert "response" in response.json()


def test_confirm_response():
    response = client.post("/confirm", json={"feedback": "That was helpful!"})
    assert response.status_code == 200
    assert response.json()["message"] == "Thanks for the feedback!"
