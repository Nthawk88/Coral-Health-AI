from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import logging
import random
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    from backend import config as cfg
except Exception:
    import config as cfg

os.makedirs(cfg.UPLOAD_FOLDER, exist_ok=True)

FILENAME_SAFE_RE = re.compile(r'[^A-Za-z0-9_.-]+')

def sanitize_filename(name: str) -> str:
    base = os.path.basename(name)
    safe = FILENAME_SAFE_RE.sub('_', base)
    return safe[:150]

def load_class_names():
    try:
        if os.path.exists(cfg.LABELS_PATH):
            with open(cfg.LABELS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                classes = data.get('classes') or []
                if classes and isinstance(classes, list):
                    logger.info(f"Loaded class names: {classes}")
                    return classes
        logger.warning("labels.json not found; falling back to default ['rusak','sehat']")
        return ['rusak', 'sehat']
    except Exception as e:
        logger.error(f"Failed to load labels.json: {e}")
        return ['rusak', 'sehat']

CLASS_NAMES = load_class_names()

try:
    if not os.path.exists(cfg.MODEL_PATH):
        logger.warning("Model file not found. Running in demo mode.")
        model = None
    else:
        model = tf.keras.models.load_model(cfg.MODEL_PATH)
        logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in cfg.ALLOWED_EXTENSIONS

def verify_image_bytes(image_data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(image_data)) as im:
            im.verify()
    except Exception as e:
        raise ValueError('Uploaded file is not a valid image') from e

def preprocess_image(image_data):
    try:
        img = Image.open(io.BytesIO(image_data))
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if img.size[0] < 100 or img.size[1] < 100:
            raise ValueError("Image dimensions too small")
        img = img.resize((224, 224))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def serve_frontend():
    return send_from_directory(cfg.FRONTEND_DIR, 'index.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'upload_folder_exists': os.path.exists(cfg.UPLOAD_FOLDER),
        'classes': CLASS_NAMES,
        'mode': 'demo' if model is None else 'production'
    })

@app.route('/version')
def version():
    return jsonify({'name': cfg.APP_NAME, 'version': cfg.APP_VERSION})

METRICS = {
    'requests_total': 0,
    'predictions_total': 0,
    'errors_total': 0
}

@app.before_request
def _count_request():
    METRICS['requests_total'] += 1

@app.route('/metrics')
def metrics():
    return jsonify(METRICS)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        METRICS['errors_total'] += 1
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file or not allowed_file(file.filename):
        METRICS['errors_total'] += 1
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG.'}), 400

    try:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > cfg.MAX_FILE_SIZE:
            METRICS['errors_total'] += 1
            return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 400

        image_data = file.read()
        verify_image_bytes(image_data)
        
        if model is None:
            health_status = random.choice(["Healthy", "Damaged"])
            confidence = random.uniform(0.6, 0.95)
        else:
            processed_image = preprocess_image(image_data)
            prediction = model.predict(processed_image, verbose=0)[0]
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])
            predicted_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class)
            logger.info(json.dumps({'predicted_label': predicted_label, 'confidence': round(confidence, 4), 'raw': [round(x, 6) for x in prediction.tolist()]}))
            METRICS['predictions_total'] += 1

            if predicted_label == 'rusak':
                health_status = "Damaged"
            else:
                health_status = "Healthy"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_name = sanitize_filename(file.filename)
        filename = f"coral_{timestamp}_{safe_name}"
        file_path = os.path.join(cfg.UPLOAD_FOLDER, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)

        return jsonify({
            'status': 'success',
            'health_status': health_status,
            'confidence': confidence,
            'filename': filename,
            'timestamp': timestamp,
            'classes': CLASS_NAMES,
            'mode': 'demo' if model is None else 'production'
        })

    except ValueError as ve:
        METRICS['errors_total'] += 1
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        METRICS['errors_total'] += 1
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image. Please try again.'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(cfg.UPLOAD_FOLDER, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    if model is None:
        logger.warning("Running in demo mode - using random predictions")
    app.run(debug=cfg.DEBUG, port=cfg.PORT)
