from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from pathlib import Path
import base64, io, time
from PIL import Image
import numpy as np
import shutil

app = FastAPI()

# Allow requests from your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train/weights/best.pt")

class ImagePayload(BaseModel):
    image: str  # Base64 string

@app.post("/predict")
async def predict(payload: ImagePayload):
    try:
        # Decode base64 → PIL image
        image_bytes = base64.b64decode(payload.image.split(",")[-1])
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Save temporary input
        timestamp = str(int(time.time() * 1000))
        input_path = Path(f"temp_input_{timestamp}.jpg")
        img.save(input_path)

        # Run YOLO
        run_dir = Path("temp_output")
        results = model.predict(
            source=str(input_path),
            save=True,
            project=str(run_dir),
            name=f"predict_{timestamp}",
            exist_ok=False
        )

        # Get YOLO output image
        out_dir = run_dir / f"predict_{timestamp}"
        out_files = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
        if not out_files:
            return JSONResponse({"labels": [], "output_image": ""})

        with open(out_files[0], "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

        labels = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                labels.append(model.names[int(box.cls[0])])

        # Clean temp files if desired

        return JSONResponse({
            "labels": labels,
            "output_image": f"data:image/jpeg;base64,{output_b64}"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
