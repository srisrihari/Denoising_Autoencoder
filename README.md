# Denoising_Autoencoder

```markdown
# Audio Denoiser

## Overview

The Audio Denoiser is a Flask web application that allows users to upload audio files, add noise, and denoise them using trained machine learning models. The application provides a user-friendly interface for processing audio files and visualizing the results.

## Features

- Upload audio files in WAV format.
- Add different types of noise (Gaussian, White, Background) to clean audio.
- Denoise audio using two models: Baseline and Tuned.
- Calculate and display Signal-to-Noise Ratio (SNR) for clean, noisy, and denoised audio.
- Visualize waveforms of audio files.
- Download processed audio files.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python.
- **TensorFlow**: An open-source library for machine learning and deep learning.
- **Librosa**: A Python package for music and audio analysis.
- **NumPy**: A library for numerical computations in Python.
- **Matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
- **SoundFile**: A library for reading and writing sound files.

## Project Structure

/home/srihari/Documents/New_Project/AI COde Editor/
│
├── flask_app.py          # Main Flask application file
├── audio_denoiser.py     # Audio processing and model training script
├── requirements.txt      # List of dependencies
├── valid_files.txt       # List of valid audio files
├── templates/            # HTML templates for the web interface
│   └── index.html        # Main HTML file for the application
└── uploads/              # Directory for uploaded audio files


## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python flask_app.py
   ```

4. Access the application in your web browser at `http://localhost:5000`.

## Usage

1. **Upload Audio**: Use the "Upload Your Own Audio" section to select and upload a WAV file.
2. **Select Noise Type**: Choose the type of noise to add to the clean audio.
3. **Select Model Type**: Choose between the Baseline and Tuned models for denoising.
4. **Process Audio**: Click the "Process Audio" button to start the processing.
5. **View Results**: The results will display the original, noisy, and denoised audio along with their SNR values.
6. **Download Processed Files**: You can download the clean, noisy, and denoised audio files.

## Model Training

The models used for denoising are trained using the `audio_denoiser.py` script. This script processes audio files, adds noise, and trains a convolutional autoencoder model for noise reduction. The trained models are saved as `denoise_baseline.h5` and `denoise_tuned.h5`.

### Training Steps

1. Prepare your clean audio files and place them in the `clean_testset_wav` directory.
2. Run the `audio_denoiser.py` script to process the audio files and train the models:

   ```bash
   python audio_denoiser.py
   ```

3. The script will generate noisy versions of the audio files and train the models, saving them for later use.

## Hgh-Level Diagram


```
graph TD
    A[Web Browser] --> B[Flask Server]
    B --> C[Audio Processing Module]
    C --> D[Machine Learning Models]
    D --> E[Processed Audio]
    
    subgraph Frontend
        A
    end
    
    subgraph Backend
        B
        C
        D
    end
    
    subgraph Data
        E
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#ffb,stroke:#333,stroke-width:2px
```

## Low-Level Diagram


```
graph TD
    A[Flask App] --> B[Route Handlers]
    B --> C[Audio Processing Functions]
    C --> D[Model Loading]
    C --> E[Audio Processing]
    E --> F[Noise Addition]
    E --> G[Denoising]
    
    subgraph Flask Routes
        B1[Index Route] --> H[Render Template]
        B2[Demo Route] --> I[Process Demo Audio]
        B3[Upload Route] --> J[Handle File Upload]
        B4[Waveform Route] --> K[Generate Waveform]
        B5[Download Route] --> L[Serve File]
    end
    
    subgraph Audio Processing
        C1[process_audio_file] --> M[Load Audio]
        C1 --> N[Standardize Length]
        C2[add_noise] --> O[Generate Noise]
        C2 --> P[Combine Audio]
        C3[denoise_audio] --> Q[Model Prediction]
    end
    
    subgraph Models
        D1[Baseline Model] --> R[Load H5 File]
        D2[Tuned Model] --> S[Load H5 File]
    end
    
    subgraph Frontend
        T[HTML Template] --> U[Bootstrap UI]
        U --> V[Audio Players]
        U --> W[File Upload]
        U --> X[Model Selection]
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style T fill:#ffb,stroke:#333,stroke-width:2px
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or find bugs, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Librosa](https://librosa.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/en/latest/)

```
