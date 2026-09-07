from pathlib import Path

CLASS_NAMES = ("overripe", "ripe", "rotten", "unripe")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_VERSION = "1.0"
RANDOM_STATE = 42
DEFAULT_PIXELS_PER_CM = 10.0
PIPELINE_IMAGE_SIZE = (224, 224)

QUALITY_GRADES = {
    "Grade A (Premium)": "Clean fruit with minimal blemish (< 5% surface spots)",
    "Grade B (Standard)": "Minor cosmetic spotting (5% - 15% surface spots)",
    "Grade C (Processing)": "Moderate senescent spotting (15% - 30% surface spots)",
    "Reject / Rotten": "Heavy lesions (> 30% surface spots) or fungal decay",
}

SAMPLE_PRESETS = {
    "Unripe Sample (Green)": "fruit_dataset/test/unripe/musa-acuminata-freshunripe-1cdf07a1-2653-11ec-a4b4-d8c4975e38aa_jpg.rf.9500aa90b98dc6f34859be314aa61dc2.jpg",
    "Ripe Sample (Yellow)": "fruit_dataset/test/ripe/musa-acuminata-banana-ad772802-394a-11ec-996c-d8c4975e38aa_jpg.rf.ac1e17757ffe79f22868f68252821f9d.jpg",
    "Overripe Sample (Spotted)": "fruit_dataset/test/overripe/musa-acuminata-mold-e18cfd23-1d0a-11ec-87d5-d8c4975e38aa_jpg.rf.175dccdf3479ad6a73213197efe00527.jpg",
    "Rotten Sample (Decayed)": "fruit_dataset/test/rotten/musa-acuminata-banana-8b73aa0b-394a-11ec-bdfe-d8c4975e38aa_jpg.rf.2ae1ee91361647a4ff44d3015a2e1be5.jpg",
}

