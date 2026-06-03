"""Central configuration for the MPN classification experiment."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
DATASET_DIR = CODE_DIR / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "çıktılar"
CLASS_NAMES = ("PV", "ET", "MF")
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)

# Baseline / default hyperparameters (paper baseline)
BASELINE_LR = 1e-3
BASELINE_DROPOUT = 0.5
BASELINE_BATCH = 32

# Transfer-learning backbone (options: "ResNet50", "DenseNet121", "MobileNetV2")
BACKBONE = "ResNet50"

# Train / val / test: first 80% vs 20% test; then 80% of train portion -> train, 20% -> val
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20  # fraction of the 80% "training pool"

# Meta-heuristic search (keep small for local runs)
META_POPULATION = 5
META_ITERATIONS = 5
META_FITNESS_EPOCHS = 4  # 3–5 epochs during search

# Final full training
FINAL_EPOCHS = 30

# Optimization bounds: LR log-uniform handled in search space; dropout continuous; batch discrete
LR_MIN = 1e-5
LR_MAX = 1e-2
DROPOUT_MIN = 0.2
DROPOUT_MAX = 0.6
BATCH_CHOICES = (16, 32)

# Reproducibility
RANDOM_SEED = 42
