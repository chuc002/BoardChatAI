import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Print startup info with environment variables
DEV_ORG_ID = os.getenv("DEV_ORG_ID", "not-set")
DEV_USER_ID = os.getenv("DEV_USER_ID", "not-set")
print(f"BoardContinuity using ORG={DEV_ORG_ID} USER={DEV_USER_ID}")

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Import routes
from routes import *

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
