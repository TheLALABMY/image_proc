from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from PIL import Image, ImageOps, ImageFilter
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

app = Flask(__name__)

# Define upload and processed folder paths
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Main page route
@app.route('/')
def index():
    return render_template('index.html')

# Pencil Sketch route
@app.route('/pencil-sketch', methods=['GET', 'POST'])
def pencil_sketch():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        
        # Convert the uploaded image to a pencil sketch
        image = Image.open(file)
        gray_image = ImageOps.grayscale(image)
        inverted_image = ImageOps.invert(gray_image)
        blurred_image = inverted_image.filter(ImageFilter.GaussianBlur(500))
        sketch_image = Image.blend(gray_image.convert("RGB"), blurred_image.convert("RGB"), -1.5)

        # Save the generated sketch image to a temporary location
        sketch_image_path = os.path.join(PROCESSED_FOLDER, 'sketch.png')
        sketch_image.save(sketch_image_path, format="PNG")

        # Pass the URL of the generated sketch image to the template
        return render_template('pencil_sketch.html', sketch_image_url=sketch_image_path)

    return render_template('pencil_sketch.html')

# Word Cloud route
@app.route('/word-cloud', methods=['GET', 'POST'])
def word_cloud():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return "No text provided"
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, relative_scaling=0.5, contour_color='yellow', contour_width=20).generate(text)
        wordcloud_image_path = os.path.join(PROCESSED_FOLDER, 'wordcloud.png')
        wordcloud.to_file(wordcloud_image_path)

        # Pass the URL of the generated word cloud image to the template
        return render_template('word_cloud.html', wordcloud_image_url=wordcloud_image_path)

    return render_template('word_cloud.html')

# Word Cloud route
@app.route('/image-editor', methods=['GET', 'POST'])
def image_editor():
    if request.method == 'POST':
        file = request.files['image']
        effect = request.form.get('effect')

        if file.filename == '':
            return jsonify({'success': False, 'message': "No selected file"})

        # Save uploaded image
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(uploaded_image_path)

        # Load image with OpenCV
        image = cv2.imread(uploaded_image_path)

        # Apply effects based on user selection
        if effect == 'grayscale':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif effect == 'blur':
            image = cv2.GaussianBlur(image, (15, 15), 0)
        elif effect == 'rotate':
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif effect == 'flip_h':
            image = cv2.flip(image, 1)  # Flip horizontally
        elif effect == 'flip_v':
            image = cv2.flip(image, 0)  # Flip vertically
        elif effect == 'edges':
            image = cv2.Canny(image, 100, 200)  # Edge detection
        elif effect == 'sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
            image = cv2.filter2D(image, -1, kernel)
        elif effect == 'sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            image = cv2.transform(image, sepia_filter)
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Save edited image
        edited_image_path = os.path.join(PROCESSED_FOLDER, 'edited_image.png')
        cv2.imwrite(edited_image_path, image)

        # Return the image URL as JSON response
        return jsonify({'success': True, 'edited_image_url': f'/{edited_image_path}'})

    return render_template('image_editor.html')

# Face Recognition route
@app.route('/face-recognition', methods=['GET', 'POST'])
def face_recognition():
    if request.method == 'POST':
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        
        # Save uploaded image
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(uploaded_image_path)

        # Load image with OpenCV for face recognition
        image = cv2.imread(uploaded_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save image with rectangles
        face_recognition_path = os.path.join(PROCESSED_FOLDER, 'face_recognition.png')
        cv2.imwrite(face_recognition_path, image)

        return render_template('face_recognition.html', face_image_url=face_recognition_path)

    return render_template('face_recognition.html')

# Pixelation route
@app.route('/pixelate', methods=['GET', 'POST'])
def pixelate():
    if request.method == 'POST':
        file = request.files['image']
        pixel_size = int(request.form.get('pixel_size', 10))  # Default to 10 if not provided

        if file.filename == '':
            return jsonify({'success': False, 'message': "No selected file"})

        # Save uploaded image
        uploaded_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(uploaded_image_path)

        # Load image with OpenCV
        image = cv2.imread(uploaded_image_path)

        # Apply pixelation
        height, width = image.shape[:2]
        small = cv2.resize(image, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

        # Save pixelated image
        pixelated_image_path = os.path.join(PROCESSED_FOLDER, 'pixelated_image.png')
        cv2.imwrite(pixelated_image_path, pixelated)

        # Return the image URL as JSON response
        return jsonify({'success': True, 'pixelated_image_url': f'/{pixelated_image_path}'})

    return render_template('pixelate.html')

if __name__ == '__main__':
    app.run(debug=True)
