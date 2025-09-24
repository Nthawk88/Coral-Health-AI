# Coral Health AI Web Application

Detect coral reef health (Healthy vs Damaged) with a simple web UI and a TensorFlow model.

## Features
- Drag & drop image upload, live analysis
- Confidence bar with explanation
- Robust backend preprocessing (RGB conversion, EXIF orientation)
- Config-driven paths in `backend/config.py`
- Reproducible class mapping via `model/labels.json`

## Folder Structure
```
Websitebio/
├─ backend/
│  ├─ app.py
│  ├─ config.py
│  └─ uploads/
├─ frontend/
│  └─ index.html
├─ model/
│  ├─ coral_model.h5           # generated after training
│  └─ labels.json              # generated after training
├─ coral_dataset/
│  ├─ sehat/
│  └─ rusak/
└─ train_model.py
```

## Requirements
- Python 3.10+
- pip

## Setup
```bash
# From project root
python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r backend/requirements.txt
```

## Training
1) Prepare dataset:
```
coral_dataset/
  sehat/
  rusak/
```
2) Run training:
```bash
python train_model.py
```
This will generate:
- `model/coral_model.h5`
- `model/labels.json` containing the ordered class names used by the model

## Run the App
```bash
# From project root
.venv\Scripts\activate
cd backend
python app.py
```
Open `http://localhost:5000`.

## How It Works
- The training script saves ordered class names to `model/labels.json`.
- The backend loads `labels.json` and uses it to map model outputs to labels, avoiding label order mismatches.
- Images are normalized to 224x224 RGB with EXIF orientation correction to match the model input.

## Tips for Better Accuracy
- Balance images between `sehat` and `rusak`.
- Add diverse lighting, angles, and environments.
- Retrain after adding new data.

## Troubleshooting
- If you see label mix-ups, retrain and ensure `labels.json` exists alongside `coral_model.h5`.
- If you see an input shape error, confirm images are RGB (backend enforces this).

## License
MIT

