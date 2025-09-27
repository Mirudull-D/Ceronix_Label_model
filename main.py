from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from pathlib import Path
import shutil, uuid, os
import serial, re, threading, time, random

# -----------------------------------------
# 🔧 FastAPI App & CORS
# -----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------
# 🖼 YOLO Model Setup
# -----------------------------------------
model = YOLO("runs/detect/train/weights/best.pt")

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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

    labels = [
        results[0].names[int(c)]
        for c in results[0].boxes.cls.cpu().numpy()
    ] if results[0].boxes else []

    return {"output_url": output_url, "labels": labels}


# -----------------------------------------
# ⚡ Serial / Mock Data Setup
# -----------------------------------------
SERIAL_PORT = "COM3"    # Change as needed
BAUD_RATE = 9600

latest_reading = {
    "supply_v": 0.0,
    "output_v": 0.0,
    "voltage_drop": 0.0,
    "current": 0.0,
    "temp": 0.0,
}

arduino_connected = False  # Track if serial is available


def create_mock_reading():
    """Generate random mock sensor data."""
    if random.random() > 0.5:  # 50% chance to create faulty data
        return {
            "supply_v": 12 + (random.random() * 0.2 - 0.1),
            "output_v": 11.5 + (random.random() * 0.3 - 0.15),
            "voltage_drop": 1.5 + random.random() * 0.5,  # BAD (>1.0V)
            "current": 0.3 + random.random() * 0.2,
            "temp": 85 + random.random() * 10,           # BAD (>70°C)
        }
    return {  # GOOD values
        "supply_v": 12 + (random.random() * 0.2 - 0.1),
        "output_v": 11.5 + (random.random() * 0.3 - 0.15),
        "voltage_drop": 0.4 + random.random() * 0.1,
        "current": 0.3 + random.random() * 0.2,
        "temp": 30 + random.random() * 5,
    }


def read_serial():
    """Continuously read from Arduino if available."""
    global latest_reading, arduino_connected
    pattern = re.compile(
        r"Supply V:\s*([\d\.]+).*?"
        r"Output V:\s*([\d\.]+).*?"
        r"Voltage Drop:\s*([\d\.]+).*?"
        r"Current:\s*([\-\d\.]+).*?"
        r"Temp:\s*([\d\.]+)",
        re.DOTALL
    )

    while True:
        try:
            print(f"Connecting to {SERIAL_PORT}...")
            with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) as ser:
                arduino_connected = True
                buffer = ""
                while True:
                    line = ser.readline().decode(errors="ignore")
                    buffer += line
                    match = pattern.search(buffer)
                    if match:
                        latest_reading = {
                            "supply_v": float(match.group(1)),
                            "output_v": float(match.group(2)),
                            "voltage_drop": float(match.group(3)),
                            "current": float(match.group(4)),
                            "temp": float(match.group(5)),
                        }
                        buffer = ""
        except serial.SerialException as e:
            print("⚠️ Arduino not found, switching to mock data...")
            arduino_connected = False
            time.sleep(2)

# Start serial thread
threading.Thread(target=read_serial, daemon=True).start()


@app.get("/l293d")
def get_l293d_data():
    """
    Returns:
      - Live Arduino data if connected
      - Mock data if Arduino is disconnected
    """
    if arduino_connected:
        return latest_reading
    else:
        return create_mock_reading()
