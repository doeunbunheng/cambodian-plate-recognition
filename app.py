import os
import sys
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Cambodian Plate Recognition",
    page_icon="🚗",
    layout="wide",
)

MODEL_PATHS = {
    "plate": MODELS_DIR / "plate_detector.pt",
}


def clear_model_cache():
    """Clear cached model resources."""
    try:
        from streamlit.runtime.caching import cache_data_key
        st.cache_resource.clear()
        return True
    except Exception:
        return False


def load_models():
    """Load models from disk. No caching to avoid stale 'missing model' state."""
    models = {}
    for name, path in MODEL_PATHS.items():
        if path.exists():
            from ultralytics import YOLO
            models[name] = YOLO(str(path))
        else:
            models[name] = None
    return models


def models_ready(models):
    """Verify all model files exist and loaded."""
    missing = []
    for name, path in MODEL_PATHS.items():
        if not path.exists() or models.get(name) is None:
            missing.append(name)
    return missing


def detect_plates(image, model):
    results = model(image, conf=0.25)
    result = results[0]
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
    return boxes


def draw_boxes(image, boxes):
    annotated = image.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"plate {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated


def main():
    st.title("🚗 Cambodian License Plate Recognition")

    st.markdown("""
    ### Test the plate detection model
    Upload an image to detect license plates. This tests the trained YOLO plate detector.
    """)

    with st.sidebar:
        st.header("🛠️ Controls")
        if st.button("🔄 Clear model cache", help="Re-load models if they were missing before"):
            if clear_model_cache():
                st.success("Cache cleared. Rerun the app.")
            else:
                st.info("Cache already cleared.")
        st.caption(f"Plate model: `{MODEL_PATHS['plate']}`")

    models = load_models()

    missing = models_ready(models)

    if missing:
        st.error("⚠️ Plate model not found. Train it first:")
        st.code(
            "cd D:\\Training_model\\cambodian-plate-recognition\\training\n"
            "python train_plate_detector.py --epochs 10 --device cpu --imgsz 416"
        )
        st.info(f"Looking for: `{MODEL_PATHS['plate']}`")
        st.info("After training, click '🔄 Clear model cache' in the sidebar, then rerun.")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],
                                help="JPG, PNG, or JPEG of a vehicle with a license plate")

    col1, col2 = st.columns(2)

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)[:, :, ::-1]  # RGB -> BGR

        with st.spinner("Detecting plates..."):
            boxes = detect_plates(img_np, models["plate"])
            annotated = draw_boxes(img_np, boxes)

        with col1:
            st.subheader("📷 Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🔍 Detected Plate")
            st.image(annotated[:, :, ::-1], use_container_width=True)

        if boxes:
            st.success(f"✅ Found {len(boxes)} plate(s)")
            for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
                st.write(f"**Plate {i+1}:** confidence {conf:.2%}, "
                         f"size {x2-x1}x{y2-y1}px")
        else:
            st.warning("No plates detected. Try a different image or lower confidence.")

    else:
        st.info("👈 Upload an image on the left to begin.")


if __name__ == "__main__":
    main()
