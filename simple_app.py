from flask import Flask, jsonify
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Basic routes for testing
@app.route('/')
def index():
    """Basic home page."""
    return """
    <html>
        <head><title>Glitch Art App - Test Mode</title></head>
        <body>
            <h1>Glitch Art App - Test Mode</h1>
            <p>This is a simplified version of the app for debugging Heroku deployment.</p>
            <p>Check <a href="/debug">debug info</a> for more details.</p>
        </body>
    </html>
    """

@app.route('/debug')
def debug():
    """Return debug information about the environment."""
    # Collect debug information
    debug_info = {
        'python_version': sys.version,
        'environment': {k: v for k, v in os.environ.items()},
        'current_dir': os.getcwd(),
        'directory_contents': os.listdir('.'),
        'python_path': sys.path,
    }
    
    # Check if glitch_art directory exists
    if os.path.exists('glitch_art'):
        debug_info['glitch_art_exists'] = True
        debug_info['glitch_art_contents'] = os.listdir('glitch_art')
    else:
        debug_info['glitch_art_exists'] = False
    
    return jsonify(debug_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 