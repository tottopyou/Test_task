from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/match", methods=["POST"])
def match_images():
    file = request.files["file"]
    # Dummy response for Task 1 (implement the logic as per the test assignment)
    return jsonify({"result": "Task 1 completed successfully!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
