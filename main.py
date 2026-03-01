from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pymongo import MongoClient
from PIL import Image
from datetime import datetime
import os
import io

app = FastAPI()

# -----------------------------
# CORS (For Flutter / ESP32)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load trained YOLO model once
# -----------------------------
model = YOLO("best.pt")

# -----------------------------
# MongoDB Connection
# -----------------------------
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    raise Exception("MONGO_URL not set in environment variables")

client = MongoClient(MONGO_URL)
db = client["smart_dog"]
collection = db["detections"]

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Smart Dog Collar AI Running"}

# -----------------------------
# ESP32 Upload + AI Detection
# -----------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    try:
        # Read image directly from memory (no local saving)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run YOLO inference
        results = model(image)

        detected_classes = []
        highest_confidence = 0.0

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                detected_classes.append(class_name)

                if confidence > highest_confidence:
                    highest_confidence = confidence

        # Decide eating status
        if "empty_bowl" in detected_classes:
            status = "Dog Finished Eating"
        elif "food_bowl" in detected_classes:
            status = "Food Still In Bowl"
        else:
            status = "No Bowl Detected"

        # Store latest result in MongoDB
        data = {
            "_id": "latest",
            "status": status,
            "confidence": highest_confidence,
            "timestamp": datetime.utcnow()
        }

        collection.update_one(
            {"_id": "latest"},
            {"$set": data},
            upsert=True
        )

        return {
            "detected_classes": detected_classes,
            "status": status,
            "confidence": highest_confidence
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Get Latest Status (Flutter App)
# -----------------------------
@app.get("/status")
async def get_status():

    latest = collection.find_one({"_id": "latest"})

    if not latest:
        return {"status": "No Data Yet"}

    return {
        "status": latest["status"],
        "confidence": latest["confidence"],
        "timestamp": latest["timestamp"]
    }