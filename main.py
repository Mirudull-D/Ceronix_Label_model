from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from pathlib import Path
import shutil, uuid, os

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train/weights/best.pt")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Serve output images
app.mount("/output", StaticFiles(directory="output"), name="output")


@app.post("/predict1")
async def predict(file: UploadFile = File(...)):
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}_{file.filename}"
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=str(input_path),
        save=True,
        project=str(OUTPUT_DIR),
        name="my_run",
        exist_ok=True
    )


    save_dir = Path(results[0].save_dir)
    output_files = list(save_dir.glob("*.jpg"))
    if not output_files:
        return JSONResponse({"error": "No output image generated"}, status_code=500)

    output_image_path = output_files[0]
    output_url = f"/output/my_run/{output_image_path.name}"


    labels = [results[0].names[int(c)]
              for c in results[0].boxes.cls.cpu().numpy()] if results[0].boxes else []

    return {"output_url": output_url, "labels": labels}
