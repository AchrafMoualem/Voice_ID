import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Files to move if they exist at project root
candidates = [
    "history_fold1.npy",
    "history_fold2.npy",
    "history_fold3.npy",
    "history_final1.npy",
    "mean.npy",
    "std.npy",
    "label_mapping.npy",
    "model_fold1.h5",
    "model_fold2.h5",
    "model_fold3.h5",
    "final_model.h5",
]

moved = []
skipped = []

for name in candidates:
    src = PROJECT_ROOT / name
    dst = MODELS_DIR / name
    if src.exists():
        try:
            shutil.move(str(src), str(dst))
            moved.append(name)
        except Exception as e:
            skipped.append((name, str(e)))

print("Migration complete")
if moved:
    print("Moved:")
    for m in moved:
        print(" -", m)
if skipped:
    print("Skipped / failed:")
    for s in skipped:
        print(" -", s[0], ":", s[1])
else:
    if not moved:
        print("No files were found to move.")
