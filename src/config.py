"""Settings shared by the whole project. Change them here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TUNING_DIR = RESULTS_DIR / "tuning"

SEED = 42

# NHTSA five-year files (by received date) used in this project.
RAW_FILES = [
    "COMPLAINTS_RECEIVED_2015-2019",
    "COMPLAINTS_RECEIVED_2020-2024",
    "COMPLAINTS_RECEIVED_2025-2026",
]
RAW_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl/{name}.zip"

# Column positions in the NHTSA flat file (it has 51 tab-separated fields and no header row).
RAW_COLUMNS = {0: "cmpl_id", 1: "odi_no", 11: "component", 16: "received", 19: "text", 20: "source", 45: "product_type"}

# Labels: the top-level component, the TOP_K most frequent ones, everything else is OTHER.
TOP_K = 14
OTHER = "OTHER"
NHTSA_CATCH_ALL = "UNKNOWN OR OTHER"   # NHTSA's own catch-all component, we count it as OTHER

# Text rules.
MIN_TEXT_CHARS = 30
SAMPLE_SIZE = 70_000      # random sample of the cleaned complaints, so the experiments run on a laptop
NEAR_DUPLICATE_SIMILARITY = 0.8   # cosine similarity of shared word triples

# Splits. "random" is stratified. "time" trains on the oldest complaints and tests on the newest.
TRAIN_SHARE = 0.70
VAL_SHARE = 0.10          # the rest (20 percent) is the test set
SPLIT_SCHEMES = ["random", "time"]

# The Groq row runs on a fixed subset of each test set, because of rate limits.
GROQ_SUBSET_SIZE = 1000

SYSTEMS = ["tfidf_lr", "embedding_lr", "finetuned_encoder", "groq_zero_shot", "zai_glm_zero_shot"]
SYSTEM_NAMES = {
    "tfidf_lr": "TF-IDF + logistic regression",
    "embedding_lr": "Sentence embeddings + logistic regression",
    "finetuned_encoder": "Fine-tuned small encoder",
    "groq_zero_shot": "Groq zero-shot (gpt-oss-20b)",
    "zai_glm_zero_shot": "Z.ai zero-shot (GLM-4.5-Flash)",
}
