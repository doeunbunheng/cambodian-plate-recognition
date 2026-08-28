import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

class Settings:
    VEHICLE_MODEL: str = str(MODELS_DIR / "vehicle_detector.pt")
    PLATE_MODEL: str = str(MODELS_DIR / "plate_detector.pt")
    OCR_DIR: str = str(MODELS_DIR / "paddleocr")
    OCR_LANG: str = "en"
    CONFIDENCE_THRESHOLD: float = 0.5
    UPLOAD_DIR: Path = BASE_DIR / "backend" / "uploads"
    PROVINCES: list = [
        "PhnomPenh", "SiemReap", "Kandal", "Battambang", "Kampot",
        "PreahSihanouk", "Kep", "Pursat", "Kratie", "Takeo",
        "KampongCham", "KampongChhnang", "KampongSpeu", "KampongThom",
        "Ratanakiri", "MundulKiri", "PreyVeng", "SvayRieng", "SiemReap",
        "TboungKhmum", "BanteayMeanchey", "OddarMeanchey", "PreahVihear",
        "StungTreng", "KohKong", "Pailin"
    ]

settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
