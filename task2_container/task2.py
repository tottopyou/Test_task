from flask import Flask, request, jsonify, send_file
import os
import subprocess
import logging
import torch
import shutil

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("task2.log")]
)

app = Flask(__name__)

RESULT_FOLDER = "./DeepSegmentor/results"

try:
    if os.path.exists(RESULT_FOLDER):
        shutil.rmtree(RESULT_FOLDER)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    logging.info("Results cleared successfully.")
except Exception as e:
    logging.error(f"Failed to clear results: {e}")

BASE_FOLDER = "./DeepSegmentor/datasets/RoadNet"
SUBFOLDERS = ["test_image", "test_segment", "test_edge", "test_centerline"]

try:
    for subfolder in SUBFOLDERS:
        subfolder_path = os.path.join(BASE_FOLDER, subfolder)
        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)
        os.makedirs(subfolder_path, exist_ok=True)
    logging.info("Datasets cleared successfully.")
except Exception as e:
    logging.error(f"Failed to clear datasets: {e}")

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    file_path = data.get("file_path")
    gpu_id = data.get("gpu_id")
    width = data.get("width")
    height = data.get("height")
    logging.info(f"Processing file: {gpu_id}", str(gpu_id))

    if not file_path or not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return jsonify({"error": "File not found."}), 400

    logging.info(f"Processing file: {file_path}")

    try:
        # Run the model script
        logging.info("Running the model script...")
        result = subprocess.run(
            ["sh", "scripts/test_roadnet.sh", str(gpu_id), str(width), str(height)],
            cwd="./DeepSegmentor",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"Script executed successfully.")
        logging.info(f"Script stdout: {result.stdout.decode()}")
        logging.info(f"Script stderr: {result.stderr.decode()}")

        # Check result file
        result_file = os.path.join(RESULT_FOLDER, "index.html")
        if not os.path.exists(result_file):
            logging.error(f"Result file not found: {result_file}")
            return jsonify({"error": "Result file not found."}), 500

        logging.info(f"Returning result file: {result_file}")
        return send_file(result_file, mimetype="text/html")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running script: {e.stderr.decode('utf-8')}")
        return jsonify({"error": "Error during model execution."}), 500
    except Exception as e:
        logging.exception(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error occurred."}), 500
    
@app.route('/gpu_availability', methods=['GET'])
def gpu_availability():
    return jsonify({"gpu_available": torch.cuda.is_available()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
