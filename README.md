# 🚗 Cambodian License Plate Recognition System

A complete license plate recognition system for Cambodian vehicles. It detects the **vehicle**, reads the **plate category/location**, and extracts the **plate number** from an image.

## 🎯 Target Output

```json
{
  "vehicle": "FamilyCar",
  "category": "PhnomPenh",
  "plate_number": "2U-5676"
}
```

## ✅ Completed Tasks

### 1. Project Setup & Structure
- Created full project directory structure for a multi-stage ML pipeline
- Backend (FastAPI), Frontend (React), Training scripts, and Streamlit test app
- Designed architecture: `YOLO Vehicle → YOLO Plate + Corners → Perspective Transform → OCR`

### 2. Dataset Analysis & Preparation
- Analyzed `D:\Training_model\Dataset` — a YOLO-format dataset of **815 Cambodian plate images**
- **Important finding:** the dataset mixes 3 label types into one file:
  - `VC` (Vehicle Category) — vehicle bounding boxes
  - `PC` (Plate Category) — province/location, e.g. `PC_PhnomPenh`
  - `PN` (Plate Number) — the registration number, e.g. `PN_2U-5676`
- Built `training/analyze_labels.py` to categorize every class by box size
- Built `training/build_datasets.py` to split into clean YOLO datasets:
  - **Plate dataset** (usable): 316 train + 79 val images, 1 class `plate`
  - **Vehicle dataset**: limited (only ~5 real vehicle boxes)
  - Province/PC data: only 14 samples (too few to train a classifier)

### 3. Trained Plate Detector Model
- Trained **YOLOv8n** on the plate dataset (CPU)
- Result: **mAP50 = 0.89**, **recall = 0.89** after ~27 epochs
- Model saved to `models/plate_detector.pt`
- Verified: detects plates at **~90% confidence** on real test images

### 4. Environment Setup
- Identified and fixed two broken environments:
  - Python 3.14 → added PyTorch, Ultralytics, Streamlit
  - Python 3.10.0b2 (broken beta) → replaced with **stable 3.10.11** via winget for PaddlePaddle compatibility
- Installed and verified: PyTorch CPU, Ultralytics 8.4, PaddlePaddle 3.3.1, PaddleOCR 3.7

### 5. Streamlit Test App
- Built `app.py` — a simple Streamlit interface to upload images and visualize plate detection
- Shows original image + detected plate bounding boxes with confidence
- Added a "Clear model cache" control and path fixes (initially had a wrong `PROJECT_ROOT` path)

### 6. FastAPI Backend
- `backend/main.py` — API server with `POST /api/recognize`
- `backend/pipeline.py` — 3-stage recognition pipeline (vehicle, plate, OCR)
- `backend/config.py` — settings + province list
- `backend/models/schemas.py` — Pydantic response models

### 7. React Frontend (full web app)
- `frontend/` — React + TypeScript + TailwindCSS
- Components: `ImageUploader`, `DetectionResults`, annotated image display
- `src/services/api.ts` — calls the FastAPI backend

### 8. Pushed to GitHub
- Repo: **https://github.com/doeunbunheng/cambodian-plate-recognition**
- 25 source files committed to `main` branch
- Model weights (`*.pt`), datasets, and training runs excluded via `.gitignore`

---

## 🏗️ Architecture

```
                     INPUT IMAGE
                          │
                          ▼
              ┌──────────────────────┐
              │  Model 1: YOLO       │
              │ Vehicle Detection    │
              └──────────────────────┘
                          │
                          ▼
              ┌──────────────────────┐
              │ Model 2: Plate       │
              │ Detection + corners  │
              │ YOLO / YOLO-Pose     │
              └──────────────────────┘
                          │
                          ▼
                    Crop Plate
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
     ┌──────────────────┐    ┌──────────────────┐
     │ Plate Category   │    │ OCR              │
     │ Classification   │    │ Plate Number     │
     └──────────────────┘    └──────────────────┘
                          │
                          ▼
                    FINAL RESULT
        Vehicle / Category / Plate Number
```

## 📁 Project Structure

```
cambodian-plate-recognition/
├── app.py               # Streamlit test app (quick testing)
├── backend/             # FastAPI server
│   ├── main.py          # API entry point
│   ├── pipeline.py      # Recognition pipeline
│   ├── config.py        # Settings + province list
│   ├── requirements.txt
│   └── models/schemas.py
├── frontend/            # React + TypeScript web app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   └── services/api.ts
│   ├── package.json
│   └── vite.config.ts
├── training/            # Dataset + training scripts
│   ├── analyze_labels.py           # Categorize VC/PC/PN by box size
│   ├── build_datasets.py           # Build clean YOLO datasets
│   ├── prepare_dataset.py
│   ├── train_plate_detector.py     # Train plate detector (YOLOv8)
│   └── train_vehicle_detector.py
├── models/              # Trained model weights (gitignored)
│   └── plate_detector.pt
├── docker-compose.yml
├── README.md
└── .gitignore
```

## 🚀 Running the Streamlit Test App

The quickest way to test plate detection:

```bash
cd D:\Training_model\cambodian-plate-recognition
streamlit run app.py
```

> ⚠️ **Important:** The app must run with **Python 3.10** (stable 3.10.11) if you use OCR/PaddlePaddle. On Python 3.14 only plate detection works (no OCR).

Open `http://localhost:8501`, upload an image, and see detected plate boxes with confidence.

## 🔧 Training Scripts

```bash
cd training

# 1. Analyze the dataset (categorize VC/PC/PN classes)
python analyze_labels.py

# 2. Build clean YOLO datasets
python build_datasets.py

# 3. Train the plate detector
python train_plate_detector.py --epochs 30 --device cpu --imgsz 640

# 4. Train the vehicle detector (needs more annotated data)
python train_vehicle_detector.py
```

## 🌐 Running the Full Web App

```bash
# Backend (FastAPI)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (React) — separate terminal
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## 🐳 Docker Deployment

```bash
docker-compose up -d
```

## 📡 API

### `POST /api/recognize`
Upload an image file. Returns:

```json
{
  "vehicle": "FamilyCar",
  "category": "PhnomPenh",
  "plate_number": "2U-5676"
}
```

### `GET /health`
Checks model availability.

---

## ⚠️ Known Limitations & Next Steps

| Area | Status | Next Step |
|------|--------|-----------|
| Plate detection | ✅ Working (mAP50 0.89) | Tune confidence / add pose corners |
| Vehicle detection | ⚠️ Needs more data | Re-annotate vehicle boxes (only 5 samples) |
| Plate category (PC) | ⚠️ Needs more data | Collect more province annotations (only 14) |
| Plate number OCR | ⚠️ In progress | Fix PaddleOCR 3.x CPU inference bug, then wire into app |
| Streamlit testing | ✅ Working | See `app.py` |

**OCR note:** PaddleOCR 3.7 + PaddlePaddle 3.3.1 has a CPU inference error
(`ConvertPirAttribute2RuntimeAttribute not supported`) with the new PP-OCRv6 models.
Options to resolve: use an older stable model, disable OneDNN/PIR, or downgrade the stack.

---

## 👤 Credits

Built for Cambodian vehicle/license plate recognition using a multi-stage YOLO + OCR pipeline.
