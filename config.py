from pathlib import Path

# --- ПУТИ К ПАПКАМ И ФАЙЛАМ ---
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Конкретные файлы данных
TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

# Папки для сохранения результатов работы нейросети
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final_model"
LOGS_DIR = BASE_DIR / "logs"

# Папка для текстовых отчетов по валидации
REPORTS_DIR = BASE_DIR / "reports"
VALIDATION_REPORT_FILE = REPORTS_DIR / "validation_report.md"

# --- НАСТРОЙКИ МОДЕЛИ И ТОКЕНИЗАЦИИ ---
MODEL_NAME = "cointegrated/rubert-tiny2"

# Количество классов
NUM_LABELS = 2

MAX_LENGTH = 128

# --- ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ---
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
SEED = 42

LABEL_MAPPING = {
    "order": 0,
    "not_order": 1
}


EVALUATE=True
