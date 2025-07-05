from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
from PIL import Image
import cv2  
from preprocessing import process_single_image


app = Flask(__name__)

# Upload foto yg dipilih ke folder static/uploads
app.config['UPLOAD_FOLDER']='static/uploads'
app.secret_key = 'some_secret_key'  # Ganti dengan string acak di produksi

# Load Model Klasifikasi
model = joblib.load('model/svm_model_klasifikasi_daun.pkl')

@app.route('/', methods=['GET', 'POST'])
@app.route("/home")
def home_page():
    prediction = session.pop('prediction', None)
    confidence = session.pop('confidence', None)
    class_probs = session.pop('class_probs', None)
    image_path = session.pop('image_path', None)
    error_message = session.pop('error_message', None)

    if request.method == 'POST':
        if 'image' not in request.files:
            session['error_message'] = "Tidak ada file yang dipilih"
            return redirect(url_for('home_page'))

        file = request.files['image']

        if file.filename == '':
            session['error_message'] = "Tidak ada file yang dipilih"
            return redirect(url_for('home_page'))

        if file:
            filename = secure_filename(file.filename)
            image_path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path_to_save)

            # Panggil preprocessing dengan 'filename' (string), bukan objek 'file'
            processed_image = process_single_image(image_path_to_save)

            if processed_image is not None:
                # Lakukan prediksi
                # Probabilitas prediksi (array of probabilities)
                probs = model.predict_proba([processed_image.flatten()])[0]  # ex: [0.1, 0.9]
                prediction_result = model.predict([processed_image.flatten()])[0]
                confidence = np.max(probs)  # nilai tertinggi dari probabilitas
                
                class_probs = dict(zip(model.classes_, probs))  # {'Mangga': 0.12, 'Jambu': 0.87, dst}
                class_probs_sorted = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)  # urut dari tertinggi
                
                # Simpan hasil ke session
                session['prediction'] = str(prediction_result) # Konversi ke string untuk session
                session['confidence'] = round(confidence * 100, 2)
                session['class_probs'] = [(label, round(prob * 100, 2)) for label, prob in class_probs_sorted]
                session['image_path'] = url_for('static', filename='uploads/' + filename)

            else:
                session['error_message'] = "Gagal memproses gambar. Pastikan file adalah gambar yang valid."

            return redirect(url_for('home_page')) # redirect setelah POST

    return render_template('home.html', active_home='active', prediction=prediction,confidence=confidence, class_probs = class_probs, image_path=image_path, error_message=error_message)

@app.route("/about")
def about_page():
    return render_template('about.html', active_about ='active')

if __name__ == "__main__":
    app.run(debug=True)