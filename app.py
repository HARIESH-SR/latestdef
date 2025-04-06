from flask import Flask, render_template, request, jsonify
from livereload import Server  # ✅ Import livereload here

import cv2
import numpy as np
import base64

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True


@app.route('/')
def frontpage():
    return render_template("frontpage.html")  # Load frontpage first

@app.route('/detect')
def index():
    return render_template("index.html")  # Redirect here when 'Start Detection' is clicked

@app.route('/analysis')
def analysis():
    return render_template("analysis.html")  # This renders analysis.html from /templates

@app.route('/detect', methods=['POST'])
def detect_deforestation():
    before_img = request.files.get("before")
    after_img = request.files.get("after")

    if not before_img or not after_img:
        return jsonify({"error": "Both images are required!"}), 400

    # Convert images to OpenCV format
    before_img_np = np.frombuffer(before_img.read(), np.uint8)
    after_img_np = np.frombuffer(after_img.read(), np.uint8)

    img_1 = cv2.imdecode(before_img_np, cv2.IMREAD_COLOR)
    img_2 = cv2.imdecode(after_img_np, cv2.IMREAD_COLOR)

    if img_1 is None or img_2 is None:
        return jsonify({"error": "Invalid image format!"}), 400

    # Resize img_2 to match img_1
    img_2_resized = cv2.resize(img_2, (img_1.shape[1], img_1.shape[0]))

    # Convert to grayscale
    gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(img_2_resized, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray_1, gray_2)

    # Apply binary threshold
    _, thresholded_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Calculate deforestation rate
    changed_pixels = np.count_nonzero(thresholded_diff)
    total_pixels = thresholded_diff.size
    deforestation_rate = round(min((changed_pixels / total_pixels) * 100, 100), 2)

    # Convert to base64 for display
    _, buffer = cv2.imencode('.png', thresholded_diff)
    diff_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "deforestation_rate": deforestation_rate,
        "image": f"data:image/png;base64,{diff_base64}"
    })

if __name__ == '__main__':
    server = Server(app.wsgi_app)
    server.watch('templates/*')  # Watches HTML files
    server.watch('static/*')     # Watches CSS/JS files
    server.serve(host='0.0.0.0', port=5000, debug=True)

