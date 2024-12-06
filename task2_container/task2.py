from flask import Flask, request, send_file
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = './DeepSegmentor/datasets/RoadNet'
RESULT_FOLDER = './DeepSegmentor/results/roadnet'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save the uploaded file
    file.save(file_path)

    try:
        # Call the test_roadnet.sh script
        subprocess.run(
            ["sh", "scripts/test_roadnet.sh","-1"],
            cwd="./DeepSegmentor",  # Ensure the script is run from the correct directory
            check=True,
        )

        # Load the output image
        output_image_path = os.path.join(RESULT_FOLDER, "output.png")  # Adjust if the script saves elsewhere
        if not os.path.exists(output_image_path):
            return {"error": "Output image not found"}, 500

        # Send the output image as a response
        return send_file(output_image_path, mimetype='image/png')

    except subprocess.CalledProcessError as e:
        # Log more information about the subprocess error
        print(f"Subprocess error: {e}")
        return {"error": f"Script execution failed: {e}"}, 500
    except Exception as e:
        # Log more information about the general error
        print(f"Unexpected error: {e}")
        return {"error": f"An error occurred: {e}"}, 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
