import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from keras import layers, models
from keras.optimizers import Adam
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Process audio files (truncate to 2 seconds)
def process_audio(input_folder, output_folder, max_files=20):
    """
    Process audio files from input folder, truncate to 2 seconds,
    and save to output folder.
    
    Args:
        input_folder: Path to folder containing input .wav files
        output_folder: Path to save processed files
        max_files: Maximum number of files to process
        
    Returns:
        List of processed filenames
    """
    os.makedirs(output_folder, exist_ok=True)
    valid_files = []
    
    for idx, file in enumerate(os.listdir(input_folder)):
        if file.endswith('.wav') and idx < max_files:
            try:
                audio, sr = librosa.load(os.path.join(input_folder, file), sr=44100, mono=True)
                if len(audio) >= 88200:  # 2 seconds at 44.1kHz
                    truncated = librosa.util.fix_length(audio, size=88200)
                    sf.write(os.path.join(output_folder, file), truncated, 44100)
                    valid_files.append(file)
                    print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Save list of valid files for later use
    with open("valid_files.txt", "w") as f:
        for file in valid_files:
            f.write(f"{file}\n")
            
    print(f"Processed {len(valid_files)} files")
    return valid_files

# 2. Add noise to clean audio
def add_noise(clean_audio, noise_type=None):
    """
    Add noise to clean audio.
    
    Args:
        clean_audio: Clean audio signal
        noise_type: Type of noise to add (gaussian, white, background)
                    If None, randomly choose one
                    
    Returns:
        Noisy audio signal
    """
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
    return noisy_audio

# 3. Build autoencoder model
def build_model(input_shape=(88200, 1)):
    """
    Build 1D convolutional autoencoder for noise reduction.
    
    Args:
        input_shape: Shape of input audio (samples, channels)
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(32, 16, padding='same', activation='relu'),
        layers.MaxPooling1D(4),
        layers.Conv1DTranspose(32, 16, padding='same', activation='relu'),
        layers.UpSampling1D(4),
        layers.Conv1D(1, 16, padding='same', activation='tanh')
    ])
    
    model.summary()
    return model

# Calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(clean, noisy):
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        clean: Clean audio signal
        noisy: Noisy audio signal
        
    Returns:
        SNR in dB
    """
    noise = clean - noisy
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# Load data from files
def load_data(file_list, folder):
    """
    Load audio data from files.
    
    Args:
        file_list: List of filenames
        folder: Folder containing audio files
        
    Returns:
        Numpy array of audio data
    """
    X = []
    for file in file_list:
        audio, _ = librosa.load(os.path.join(folder, file), sr=44100)
        X.append(audio.reshape(-1, 1))
    return np.array(X)

# Plot waveforms for comparison
def plot_waveforms(clean, noisy, denoised, title):
    """
    Plot waveforms for comparison.
    
    Args:
        clean: Clean audio signal
        noisy: Noisy audio signal
        denoised: Denoised audio signal
        title: Title for the plot
    """
    plt.figure(figsize=(15, 9))
    
    plt.subplot(3, 1, 1)
    plt.plot(clean)
    plt.title('Clean Audio')
    plt.xlim(0, len(clean))
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy)
    plt.title(f'Noisy Audio (SNR: {calculate_snr(clean, noisy):.2f} dB)')
    plt.xlim(0, len(noisy))
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised)
    plt.title(f'Denoised Audio (SNR: {calculate_snr(clean, denoised):.2f} dB)')
    plt.xlim(0, len(denoised))
    
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()

# Main execution
def main():
    # Directories
    input_folder = "clean_testset_wav"
    clean_folder = "processed_clean"
    noisy_folder = "processed_noisy"
    denoised_baseline_folder = "denoised_baseline"
    denoised_tuned_folder = "denoised_tuned"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist. Creating an empty folder.")
        os.makedirs(input_folder, exist_ok=True)
        print(f"Please place your .wav files in the '{input_folder}' directory and run the script again.")
        return
    
    # 1. Process clean audio files
    valid_files = process_audio(input_folder, clean_folder, max_files=20)
    
    if not valid_files:
        print("No valid files found. Please add .wav files with at least 2 seconds duration to the input folder.")
        return
    
    # 2. Generate noisy versions
    os.makedirs(noisy_folder, exist_ok=True)
    for file in valid_files:
        clean, sr = librosa.load(os.path.join(clean_folder, file), sr=44100)
        noisy = add_noise(clean)
        sf.write(os.path.join(noisy_folder, file), noisy, sr)
    print(f"Generated noisy versions for {len(valid_files)} files")
    
    # 3. Load data for training
    X_clean = load_data(valid_files, clean_folder)
    X_noisy = load_data(valid_files, noisy_folder)
    
    # 4. Baseline model training
    print("\n=== Baseline Model Training ===")
    model = build_model()
    model.compile(optimizer='adam', loss='mse')
    
    history = model.fit(
        X_noisy, X_clean,
        epochs=30,
        batch_size=min(32, len(valid_files)),
        validation_split=0.2,
        verbose=1
    )
    
    # Save baseline model
    model.save("denoise_baseline.h5")
    print("Baseline model saved as 'denoise_baseline.h5'")
    
    # Generate and save denoised outputs using baseline model
    os.makedirs(denoised_baseline_folder, exist_ok=True)
    for idx, file in enumerate(valid_files):
        denoised = model.predict(X_noisy[idx].reshape(1, 88200, 1))
        sf.write(os.path.join(denoised_baseline_folder, file), denoised.flatten(), 44100)
        
        # Plot comparison for the first file
        if idx == 0:
            plot_waveforms(
                X_clean[idx].flatten(),
                X_noisy[idx].flatten(),
                denoised.flatten(),
                "baseline_comparison"
            )
    
    print(f"Saved denoised outputs to '{denoised_baseline_folder}'")
    
    # 5. Hyperparameter tuning
    print("\n=== Hyperparameter Tuning ===")
    learning_rates = [0.001, 0.0005]
    best_loss = float('inf')
    best_model = None
    
    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        model = build_model()
        model.compile(optimizer=Adam(lr), loss='mse')
        
        history = model.fit(
            X_noisy, X_clean,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        val_loss = history.history['val_loss'][-1]
        print(f"Validation loss: {val_loss}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
    
    # Save tuned model
    best_model.save("denoise_tuned.h5")
    print(f"Tuned model saved as 'denoise_tuned.h5' (best validation loss: {best_loss})")
    
    # Generate and save denoised outputs using tuned model
    os.makedirs(denoised_tuned_folder, exist_ok=True)
    for idx, file in enumerate(valid_files):
        denoised = best_model.predict(X_noisy[idx].reshape(1, 88200, 1))
        sf.write(os.path.join(denoised_tuned_folder, file), denoised.flatten(), 44100)
        
        # Plot comparison for the first file
        if idx == 0:
            plot_waveforms(
                X_clean[idx].flatten(),
                X_noisy[idx].flatten(),
                denoised.flatten(),
                "tuned_comparison"
            )
    
    print(f"Saved denoised outputs to '{denoised_tuned_folder}'")
    
    # 6. Print summary
    print("\n=== Project Summary ===")
    print(f"- Processed {len(valid_files)} audio files")
    print(f"- Created folders: {clean_folder}, {noisy_folder}, {denoised_baseline_folder}, {denoised_tuned_folder}")
    print(f"- Saved models: denoise_baseline.h5, denoise_tuned.h5")
    print(f"- Saved list of valid files: valid_files.txt")

if __name__ == "__main__":
    main()