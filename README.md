# Ceronix Label Model

A FastAPI-based application that uses **YOLOv8** for detecting labels on **Ceronix displays**. The project provides an API interface for image-based inference, making it easy to integrate with other services.

---

## Introduction
The **Ceronix Label Model** is built to automate detection of labels from images of Ceronix displays.  
It leverages **YOLOv8** for real-time object detection and provides results through a **FastAPI** server.

---

## Features
- REST API powered by FastAPI.  
- Real-time label detection using YOLOv8.  
- Easy-to-use endpoint for image uploads.  
- Sample test images included for demonstration.  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mirudull-D/Ceronix_Label_model.git
   cd Ceronix_Label_model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Run FastAPI Server
```bash
uvicorn main:app --reload
```

Server will start at:  
👉 http://127.0.0.1:8000  

### Send an Image for Detection
Using `cURL`:
```bash
curl -X POST "http://127.0.0.1:8000/detect"      -F "file=@test.jpg"
```

Results will be saved under `runs/detect/exp/`.

---

## API Endpoints

| Method | Endpoint   | Description |
|--------|-----------|-------------|
| `POST` | `/detect` | Upload image and run YOLOv8 detection |

---

## Dependencies
Main dependencies (see `requirements.txt` for full list):
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Uvicorn](https://www.uvicorn.org/)  
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)  
- [Pillow](https://pypi.org/project/Pillow/)  

---

## Configuration
- **Model Weights**: YOLOv8 weights are loaded in `main.py`. Update this if using custom weights.  
- **Runs Directory**: Detection results are stored in `runs/detect/`.  
- **Port**: Default is `8000`. You can change it with the `uvicorn` command:  
  ```bash
  uvicorn main:app --host 0.0.0.0 --port 9000
  ```

---

## Examples
- Input: `test.jpg`  
- Output: Detection results in `runs/detect/exp/`  

---

## Troubleshooting
- **Error:** `ModuleNotFoundError: No module named 'ultralytics'`  
  ✅ Run `pip install ultralytics`.  

- **Error:** Server not starting with `uvicorn`.  
  ✅ Install with `pip install uvicorn`.  

- **Error:** Permission issues writing to `runs/detect/`.  
  ✅ Check folder permissions or run with admin rights.  

---

## Contributors
- [Mirudull D](https://github.com/Mirudull-D)  

---

## License
This project is licensed under the **MIT License**.  
