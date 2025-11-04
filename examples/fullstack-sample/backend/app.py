from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes, allowing frontend to call backend

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/api/greeting', methods=['GET'])
def get_greeting():
    return jsonify(message="Hello from the Python backend!")

if __name__ == '__main__':
    # Note: `flask run` is the recommended way to start the dev server.
    # This `if __name__ == '__main__':` block is for direct execution `python app.py`
    # which is simpler for some environments but `flask run` offers more features.
    app.run(debug=True, port=5000)
