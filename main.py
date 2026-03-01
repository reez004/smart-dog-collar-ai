from fastapi import FastAPI, UploadFile, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from pymongo import MongoClient
from PIL import Image
from datetime import datetime
from typing import List
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
# Load YOLO Model Once
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
# WebSocket Clients Storage
# -----------------------------
connected_clients: List[WebSocket] = []

# -----------------------------
# Home Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Smart Dog Collar AI Running"}

# -----------------------------
# WebSocket Endpoint
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            await websocket.receive_text()
    except:
        connected_clients.remove(websocket)

# -----------------------------
# ESP32 Upload + AI Detection
# -----------------------------
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

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

        # Data to store
        data = {
            "_id": "latest",
            "status": status,
            "confidence": highest_confidence,
            "timestamp": datetime.utcnow()
        }

        # Save in MongoDB
        collection.update_one(
            {"_id": "latest"},
            {"$set": data},
            upsert=True
        )

        # Push to WebSocket clients (REAL-TIME)
        for client in connected_clients:
            await client.send_json({
                "status": status,
                "confidence": highest_confidence,
                "timestamp": str(datetime.utcnow())
            })

        return {
            "detected_classes": detected_classes,
            "status": status,
            "confidence": highest_confidence
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Get Latest Status (Flutter Fallback)
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