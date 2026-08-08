from pathlib import Path

CLASS_NAMES = ("overripe", "ripe", "rotten", "unripe")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FEATURE_VERSION = "1.0"
RANDOM_STATE = 42

