import pytest
import sys
import os

# Add the parent directory to the Python path to allow importing 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app as flask_app

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_hello_api(client):
    """
    Test the /api/hello endpoint.
    """
    response = client.get('/api/hello')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data == {"message": "Hello from Flask API!"}
