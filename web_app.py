"""
Web Application - Uses CLI's HKBUAssistant class directly
This ensures 100% consistency with CLI behavior
"""
from src.config import load_config
from src.cli.main import HKBUAssistant
import os
import sys
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from CLI module

app = Flask(__name__, static_folder='web_ui/build', static_url_path='')
CORS(app)

# Load config
config = load_config()

# Store assistant instances per session
assistants = {}


def get_or_create_assistant(session_id):
    """Get existing assistant or create new one."""
    if session_id not in assistants:
        print(f"Creating new assistant for session: {session_id[:8]}...")
        assistant = HKBUAssistant(config)
        assistant.session_id = session_id
        assistants[session_id] = assistant
    return assistants[session_id]


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'mode': 'cli_class_wrapper',
        'active_sessions': len(assistants)
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint using CLI's HKBUAssistant."""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', str(uuid.uuid4()))
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 512)

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        print(f"[{session_id[:8]}] User: {message}")
        print(
            f"[{session_id[:8]}] Params: temp={temperature}, max_tokens={max_tokens}")

        # Get or create assistant for this session
        assistant = get_or_create_assistant(session_id)

        # Update chat service parameters
        if assistant.chat_service:
            assistant.chat_service.temperature = temperature
            assistant.chat_service.max_tokens = max_tokens

        # Use the assistant's chat method directly
        response = assistant.chat(message)

        print(f"[{session_id[:8]}] Assistant: {response[:100]}...")

        # Determine response type
        response_type = 'chat'
        if assistant.study_plan_manager:
            if assistant.study_plan_manager.is_study_plan_query(message):
                response_type = 'study_plan'
            elif assistant.study_plan_manager.conversation_state == 'collecting_constraints':
                response_type = 'study_plan'

        return jsonify({
            'response': response,
            'session_id': session_id,
            'type': response_type
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'response': f"Error: {str(e)}",
            'session_id': session_id,
            'type': 'error'
        }), 500


@app.route('/api/new-session', methods=['POST'])
def new_session():
    """Create a new session."""
    session_id = str(uuid.uuid4())

    # Pre-create assistant
    try:
        get_or_create_assistant(session_id)
        print(f"New session created: {session_id[:8]}...")
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End a session."""
    data = request.json
    session_id = data.get('session_id')

    if session_id in assistants:
        del assistants[session_id]
        print(f"Session ended: {session_id[:8]}...")

    return jsonify({'status': 'ended'})

# Serve React app


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve the React frontend."""
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    print("=" * 60)
    print("HKBU Assistant Web Server (CLI Class Wrapper)")
    print("=" * 60)
    print("\nThis server uses the CLI's HKBUAssistant class directly.")
    print("Guarantees 100% consistency with CLI behavior.\n")

    port = int(os.environ.get('PORT', 5001))
    print(f"Starting server on http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
