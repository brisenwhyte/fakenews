from flask import Flask, request, jsonify
from flask_cors import CORS
import article_analyzer  # Import the refactored logic

# Initialize Flask app
app = Flask(__name__)

# Enable CORS to allow requests from your React app (http://localhost:3000)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

@app.route('/api/analyze-article', methods=['POST'])
def analyze_article_endpoint():
    """
    API endpoint to analyze an article from a URL or pasted text.
    Expects a JSON body with either a "url" or "text" key.
    """
    # Check if the request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Validate input
    if not data or ('url' not in data and 'text' not in data):
        return jsonify({"error": "Missing 'url' or 'text' in request body"}), 400
    
    if data.get('url') and not data.get('url').strip():
         return jsonify({"error": "'url' cannot be empty"}), 400
    
    if data.get('text') and not data.get('text').strip():
         return jsonify({"error": "'text' cannot be empty"}), 400

    print(f"Received request: {data}")

    try:
        # Call the main analysis function from our logic file
        result = article_analyzer.run_analysis(data)
        print(f"Analysis complete. Sending result: {result}")
        return jsonify(result)
    except Exception as e:
        # Log the full error for debugging on the server
        print(f"An unexpected error occurred: {e}")
        # Return a generic error to the client
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # Run the app on port 5001 to avoid conflicts with other services
    # Host '0.0.0.0' makes it accessible from your local network
    app.run(host='0.0.0.0', port=5001, debug=True)