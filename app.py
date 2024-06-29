from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        return None, "Error in stitching images"
    stitched_image = Image.fromarray(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
    return stitched_image, None

def crop_missing_parts(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped_image = image_cv[y:y+h, x:x+w]
    else:
        cropped_image = image_cv
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    return cropped_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stitch', methods=['POST'])
def stitch():
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "No images uploaded"}), 400
    images = [cv2.cvtColor(np.array(Image.open(file.stream)), cv2.COLOR_RGB2BGR) for file in files]
    stitched_image, error = stitch_images(images)
    if error:
        return jsonify({"error": error}), 400
    cropped_image = crop_missing_parts(stitched_image)
    stitched_bytes = io.BytesIO()
    stitched_image.save(stitched_bytes, format='JPEG')
    stitched_bytes.seek(0)
    cropped_bytes = io.BytesIO()
    cropped_image.save(cropped_bytes, format='JPEG')
    cropped_bytes.seek(0)
    return send_file(io.BytesIO(stitched_bytes.read() + cropped_bytes.read()), mimetype='application/octet-stream')

if __name__ == '__main__':
    app.run(debug=True)
