from flask import Flask, request, render_template, send_file, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)


model = tf.keras.models.load_model("minecraft-skin-optimizer.h5")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-image", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify("No file part"), 400
    
    file = request.files["file"]
    
    if not file:
        return jsonify("No selected file"), 400
        
    if not file.filename.endswith(".png"):
        return jsonify("Invalid file type"), 400
        
    image = Image.open(file).convert("RGBA")
        
    if image.size != (64, 64):
        return "Image must be 64x64 pixels", 400
        
    normalized_image_array = np.array(image) / 255.0
    
    input_array = np.array([normalized_image_array])
    predictions = model.predict(input_array)
        
    output_array = (predictions[0] * 255.0).astype(np.uint8)
    output_image = Image.fromarray(output_array, "RGBA")
        
    output_byte_arr = io.BytesIO()
    output_image.save(output_byte_arr, format="PNG")
    output_byte_arr.seek(0)
        
    return send_file(output_byte_arr, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
