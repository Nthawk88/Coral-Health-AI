import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

MODEL_PATH = os.path.join(MODEL_DIR, 'coral_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024

DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
PORT = int(os.getenv('PORT', '5000'))

APP_NAME = 'Coral Health AI'
APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
