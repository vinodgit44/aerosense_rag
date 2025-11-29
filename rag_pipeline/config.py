from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parents[1]

@dataclass
class Paths:
    data_dir: Path = BASE_DIR / "data"
    manuals_dir: Path = data_dir / "manuals"
    logs_dir: Path = data_dir / "logs"
    ground_truth_dir: Path = data_dir / "ground_truth"
    vector_db_dir: Path = BASE_DIR / "chroma_db"

@dataclass
class Models:
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_model: str = "tinyllama"   # or "mistral" if your PC can handle it

@dataclass
class RetrievalConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 6
    manual_weight: float = 0.6
    log_weight: float = 0.4

paths = Paths()
models = Models()
retrieval_cfg = RetrievalConfig()
