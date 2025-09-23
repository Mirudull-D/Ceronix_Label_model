import os
import shutil
from pathlib import Path

# 📁 Change these paths
src_dir = r"dataset/train/images"       # folder where your images are
out_dir = r"classified_by_name"         # output folder

# Create destination folders
categories = ["bad", "clr", "partclr"]
for cat in categories:
    Path(out_dir, cat).mkdir(parents=True, exist_ok=True)

# Loop through all images
for file in os.listdir(src_dir):
    lower_name = file.lower()
    if lower_name.startswith("bad"):
        shutil.copy(os.path.join(src_dir, file), os.path.join(out_dir, "bad", file))
    elif lower_name.startswith("clr"):
        shutil.copy(os.path.join(src_dir, file), os.path.join(out_dir, "clr", file))
    elif lower_name.startswith("partclr"):
        shutil.copy(os.path.join(src_dir, file), os.path.join(out_dir, "partclr", file))
    else:
        print(f"Skipped (no category): {file}")

print("✅ Images copied into:", out_dir)
