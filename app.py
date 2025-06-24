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
                prediction_result = model.predict([processed_image.flatten()])[0]
                
                # Simpan hasil ke session
                session['prediction'] = str(prediction_result) # Konversi ke string untuk session
                session['image_path'] = url_for('static', filename='uploads/' + filename)

            else:
                session['error_message'] = "Gagal memproses gambar. Pastikan file adalah gambar yang valid."

            return redirect(url_for('home_page')) # redirect setelah POST

    return render_template('home.html', active_home='active', prediction=prediction, image_path=image_path, error_message=error_message)

@app.route("/about")
def about_page():
    return render_template('about.html', active_about ='active')

if __name__ == "__main__":
    app.run(debug=True)