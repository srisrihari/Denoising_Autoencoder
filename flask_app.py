from flask import Flask, request, render_template, jsonify, send_file
import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import io
from werkzeug.utils import secure_filename
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)

# Load the trained models
mse = MeanSquaredError()
baseline_model = tf.keras.models.load_model('denoise_baseline.h5', custom_objects={'mse': mse})
tuned_model = tf.keras.models.load_model('denoise_tuned.h5', custom_objects={'mse': mse})

# Load list of valid files
with open('valid_files.txt', 'r') as f:
    valid_files = [line.strip() for line in f.readlines()]

# Create upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(clean, noisy):
    noise = clean - noisy
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Add noise to clean audio
def add_noise(clean_audio, noise_type=None):
    noise_types = ['gaussian', 'white', 'background']
    if noise_type is None:
        noise_type = np.random.choice(noise_types)
        
    if noise_type == 'gaussian':
        noise = 0.1 * np.random.normal(0, 1, len(clean_audio))
    elif noise_type == 'white':
        noise = 0.1 * np.random.uniform(-1, 1, len(clean_audio))
    elif noise_type == 'background':
        noise = 0.05 * np.random.randn(len(clean_audio))
    
    noisy_audio = np.clip(clean_audio + noise, -1.0, 1.0)
    return noisy_audio, noise, noise_type

# Process audio file
def process_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=44100, mono=True)
    
    # Truncate or pad to exactly 2 seconds
    if len(audio) > 88200:
        audio = audio[:88200]
    elif len(audio) < 88200:
        audio = librosa.util.fix_length(audio, size=88200)
    
    return audio, sr

# Denoise audio
def denoise_audio(audio, model_type='tuned'):
    audio_input = audio.reshape(1, 88200, 1)
    
    if model_type == 'baseline':
        denoised = baseline_model.predict(audio_input)
    else:
        denoised = tuned_model.predict(audio_input)
    
    return denoised.flatten()

@app.route('/')
def index():
    return render_template('index.html', valid_files=valid_files)

@app.route('/demo/<filename>')
def demo(filename):
    if filename not in valid_files:
        return jsonify({'error': 'File not found'}), 404
    
    # Get clean audio
    clean_path = os.path.join('processed_clean', filename)
    clean, sr = process_audio_file(clean_path)
    
    # Get noisy audio
    noisy_path = os.path.join('processed_noisy', filename)
    noisy, _ = process_audio_file(noisy_path)
    
    # Get denoised audio (baseline)
    baseline_path = os.path.join('denoised_baseline', filename)
    baseline, _ = process_audio_file(baseline_path)
    
    # Get denoised audio (tuned)
    tuned_path = os.path.join('denoised_tuned', filename)
    tuned, _ = process_audio_file(tuned_path)
    
    # Calculate SNR
    noisy_snr = calculate_snr(clean, noisy)
    baseline_snr = calculate_snr(clean, baseline)
    tuned_snr = calculate_snr(clean, tuned)
    
    return jsonify({
        'filename': filename,
        'noisy_snr': float(noisy_snr),
        'baseline_snr': float(baseline_snr),
        'tuned_snr': float(tuned_snr),
        'clean_url': f'/audio/clean/{filename}',
        'noisy_url': f'/audio/noisy/{filename}',
        'baseline_url': f'/audio/baseline/{filename}',
        'tuned_url': f'/audio/tuned/{filename}'
    })

@app.route('/audio/<folder>/<filename>')
def get_audio(folder, filename):
    if folder == 'clean':
        path = os.path.join('processed_clean', filename)
    elif folder == 'noisy':
        path = os.path.join('processed_noisy', filename)
    elif folder == 'baseline':
        path = os.path.join('denoised_baseline', filename)
    elif folder == 'tuned':
        path = os.path.join('denoised_tuned', filename)
    else:
        return jsonify({'error': 'Invalid folder'}), 400
    
    return send_file(path, mimetype='audio/wav')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    noise_type = request.form.get('noise_type', None)
    model_type = request.form.get('model_type', 'tuned')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process audio
        try:
            clean, sr = process_audio_file(file_path)
            
            # Add noise
            noisy, noise, noise_type_used = add_noise(clean, noise_type)
            
            # Denoise
            denoised = denoise_audio(noisy, model_type)
            
            # Calculate SNR
            noisy_snr = calculate_snr(clean, noisy)
            denoised_snr = calculate_snr(clean, denoised)
            
            # Save files
            clean_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clean_' + filename)
            noisy_path = os.path.join(app.config['UPLOAD_FOLDER'], 'noisy_' + filename)
            denoised_path = os.path.join(app.config['UPLOAD_FOLDER'], 'denoised_' + filename)
            
            sf.write(clean_path, clean, sr)
            sf.write(noisy_path, noisy, sr)
            sf.write(denoised_path, denoised, sr)
            
            return jsonify({
                'filename': filename,
                'noise_type': noise_type_used,
                'model_type': model_type,
                'noisy_snr': float(noisy_snr),
                'denoised_snr': float(denoised_snr),
                'clean_url': f'/download/clean_{filename}',
                'noisy_url': f'/download/noisy_{filename}',
                'denoised_url': f'/download/denoised_{filename}'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                     mimetype='audio/wav', 
                     as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)