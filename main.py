from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# Load trained model once
model = YOLO("best.pt")

UPLOAD_FOLDER = "received"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Smart Dog Collar AI Running"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO inference
    results = model(file_path)

    detected_class = None

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            detected_class = class_name

    if detected_class == "empty_bowl":
        status = "Dog Finished Eating"
    elif detected_class == "food_bowl":
        status = "Food Still In Bowl"
    else:
        status = "No Bowl Detected"

    return {"detected": detected_class, "status": status}