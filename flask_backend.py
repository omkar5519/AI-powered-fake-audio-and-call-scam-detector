from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import datetime
import os

app = Flask(__name__)
CORS(app)

# ======================
# 🔗 MongoDB Atlas Setup
# ======================
# Replace <username> and <password> with your Atlas credentials
# Example: mongodb+srv://omkar123:MyPass@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGO_URI = "mongodb+srv://omkarpote5519:omkar937@omkar.ftqyedv.mongodb.net/?appName=omkar"

try:
    client = MongoClient(MONGO_URI)
    db = client["voice_detection_db"]        # Database name
    collection = db["predictions"]           # Collection name
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print("❌ Failed to connect to MongoDB Atlas:", e)
    collection = None


# ======================
# 🏠 Root Route
# ======================
@app.route("/")
def home():
    return jsonify({"message": "MongoDB Flask backend is running!"})


# ======================
# 💾 Save Prediction API
# ======================
@app.route("/save_prediction", methods=["POST"])
def save_prediction():
    if collection is None:
        return jsonify({"error": "MongoDB connection not available"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Add timestamp before saving
        data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert into MongoDB
        collection.insert_one(data)
        print(f"📦 Saved to MongoDB: {data['filename']}")

        return jsonify({"message": "Prediction saved to MongoDB successfully!"}), 201

    except Exception as e:
        print("❌ Error saving to MongoDB:", e)
        return jsonify({"error": str(e)}), 500


# ======================
# 📜 Fetch All Predictions
# ======================
@app.route("/get_predictions", methods=["GET"])
def get_predictions():
    if collection is None:
        return jsonify({"error": "MongoDB connection not available"}), 500

    try:
        predictions = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id field
        return jsonify(predictions), 200
    except Exception as e:
        print("❌ Error fetching predictions:", e)
        return jsonify({"error": str(e)}), 500


# ======================
# 🚀 Run Server
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
