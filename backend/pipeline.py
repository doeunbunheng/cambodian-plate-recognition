import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from config import settings


class RecognitionPipeline:
    def __init__(self):
        self.vehicle_model = None
        self.plate_model = None
        self.ocr = None
        self._load_models()

    def _load_models(self):
        print("Loading models...")

        if os.path.exists(settings.VEHICLE_MODEL):
            self.vehicle_model = YOLO(settings.VEHICLE_MODEL)
        else:
            print(f"Warning: Vehicle model not found at {settings.VEHICLE_MODEL}")

        if os.path.exists(settings.PLATE_MODEL):
            self.plate_model = YOLO(settings.PLATE_MODEL)
        else:
            print(f"Warning: Plate model not found at {settings.PLATE_MODEL}")

        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=settings.OCR_LANG
            )
        except Exception as e:
            print(f"Warning: PaddleOCR failed to load: {e}")
            self.ocr = None

        print("Models loaded.")

    def predict(self, image: np.ndarray) -> dict:
        vehicle = self._detect_vehicle(image)
        plate_crop = self._detect_plate(image)
        category = ""
        plate_number = ""

        if plate_crop is not None:
            category, plate_number = self._read_plate(plate_crop)

        return {
            "vehicle": vehicle,
            "category": category,
            "plate_number": plate_number
        }

    def _detect_vehicle(self, image: np.ndarray) -> str:
        if self.vehicle_model is None:
            return "Unknown"

        results = self.vehicle_model(image, conf=settings.CONFIDENCE_THRESHOLD)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            box = result.boxes[0]
            class_id = int(box.cls[0])
            return result.names[class_id]

        # Fallback: use classification if available
        if result.probs is not None:
            top1 = int(result.probs.top1)
            return result.names[top1]

        return "Unknown"

    def _detect_plate(self, image: np.ndarray) -> np.ndarray:
        if self.plate_model is None:
            return None

        results = self.plate_model(image, conf=settings.CONFIDENCE_THRESHOLD)
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        if result.keypoints is not None and len(result.keypoints) > 0:
            keypoints = result.keypoints[0].xy[0].cpu().numpy()
            return self._warp_plate(image, keypoints)

        # Fallback: simple crop from bounding box
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
        return image[int(y1):int(y2), int(x1):int(x2)]

    def _warp_plate(self, image: np.ndarray, points: np.ndarray) -> np.ndarray:
        if len(points) < 4:
            return None

        src = np.array(points[:4], dtype=np.float32)

        width = int(np.linalg.norm(src[0] - src[1]))
        height = int(np.linalg.norm(src[1] - src[2]))

        if width == 0 or height == 0:
            return None

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        return warped

    def _read_plate(self, plate_crop: np.ndarray) -> tuple:
        if self.ocr is None:
            return "", ""

        enhanced = self._enhance_plate(plate_crop)

        try:
            result = self.ocr.ocr(enhanced, cls=True)
            texts = []
            for line in result:
                if line:
                    for item in line:
                        texts.append(item[1][0])
        except Exception:
            return "", ""

        category = self._extract_category(texts)
        plate_number = self._extract_number(texts)

        return category, plate_number

    def _enhance_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _extract_category(self, texts: list) -> str:
        for text in texts:
            clean = re.sub(r'[^a-zA-Z]', '', text.lower())
            for province in settings.PROVINCES:
                if province.lower() in clean:
                    return province
                if clean in province.lower():
                    return province
        return ""

    def _extract_number(self, texts: list) -> str:
        for text in texts:
            clean = text.replace(" ", "").replace("-", "")
            if any(c.isdigit() for c in clean) and len(clean) >= 3:
                return text.strip()
        return ""
