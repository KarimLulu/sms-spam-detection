from pathlib import Path

repo_dir = Path(__file__).resolve().parents[1]
data_dir = repo_dir / "data"
models_dir = data_dir / "models"

model_extension = ".model"

model_id = "current_model"
THRESHOLD = 0.5
SPAM_LABEL = "spam"
HAM_LABEL = "ham"
ROUND = 1
