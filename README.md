# Cambodian License Plate Recognition System

A web application that recognizes Cambodian license plates and outputs:
- **Vehicle**: Type of vehicle (FamilyCar, Truck, etc.)
- **Category**: Plate category/location (PhnomPenh, SiemReap, etc.)
- **Plate Number**: The registration number (2U-5676)

## Architecture

```
INPUT IMAGE → YOLO Vehicle Detection → YOLO Plate Detection → PaddleOCR → Final Result
```

## Tech Stack

- **Backend**: FastAPI + YOLOv8 + PaddleOCR + OpenCV
- **Frontend**: React + TypeScript + TailwindCSS
- **Deployment**: Docker

## Project Structure

```
cambodian-plate-recognition/
├── backend/           # FastAPI server
│   ├── main.py        # API entry point
│   ├── pipeline.py    # Recognition pipeline
│   ├── config.py      # Settings
│   └── models/        # Pydantic schemas
├── frontend/          # React web app
├── training/          # Training scripts
├── models/            # Trained model weights
└── Dataset/           # Your dataset
```

## Quick Start

### 1. Train models (optional - use pretrained)

```bash
cd training
python prepare_dataset.py          # Split dataset
python train_vehicle_detector.py   # Train vehicle detector
python train_plate_detector.py     # Train plate detector
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Docker Deployment

```bash
docker-compose up -d
```

## API

### POST /api/recognize
Upload an image file

**Response:**
```json
{
  "vehicle": "FamilyCar",
  "category": "PhnomPenh",
  "plate_number": "2U-5676"
}
```
