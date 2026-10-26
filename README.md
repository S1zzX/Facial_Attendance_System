# Facial Attendance System

A desktop attendance application that identifies students from a webcam and records their check-in and check-out times. The project combines a Tkinter interface, OpenCV face detection, TensorFlow/Keras recognition models, optional liveness checks, and an SQLite database.

## Features

- Live webcam face detection and recognition
- Student registration with a name, student ID, email, and face image
- Automatic daily check-in and check-out
- SQLite storage for students and attendance records
- Attendance reports by date
- Separate face-identification view with a daily text log
- Optional liveness checks using eye, smile, texture, and rPPG-based signals
- Multiple recognition models:
  - Euclidean Siamese network
  - Cosine Siamese network
  - Embedding classification model
- Model and liveness settings in the GUI

## Project Structure

```text
Face_Attendance_System/
├── main.py                         # Application entry point
├── integrated_attendance_gui.py    # Tkinter interface and application workflow
├── face_recognition_module.py      # Detection, recognition, and liveness logic
├── database.py                     # SQLite database operations
├── trainer.ipynb                   # Model training and evaluation notebook
├── requirements.txt                # Python dependencies
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   ├── siamese_euclidean.keras
│   ├── siamese_cosine.keras
│   └── embedding_classification.keras
└── faces/
    └── face_encodings.pkl          # Registered face encodings
```

The application creates `attendance.db` and identification log files at runtime.

## Requirements

- Python 3.9 or newer
- A webcam
- Windows, Linux, or macOS
- A working Tkinter installation
- GPU support is optional; TensorFlow can run on the CPU

> On Windows, installing `dlib` may require CMake and Microsoft C++ Build Tools.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/S1zzX/Facial_Attendance_System.git
   cd Facial_Attendance_System
   ```

2. Create and activate a virtual environment:

   **Windows**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Linux/macOS**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Application

Run the command from the directory containing `main.py`:

```bash
python main.py
```

The application uses webcam index `0` by default. Close other programs that may be using the camera before starting it.

## First-Time Use

1. Open the **Register Student** tab.
2. Enter the student's name, student ID, and optional email.
3. Capture a clear, front-facing image.
4. Open **Take Attendance** and start the camera.
5. The first successful recognition records a check-in. A later recognition on the same day records the check-out.
6. Use the **Reports** tab to view attendance for a selected date.

Registered face labels follow the format `Name_StudentID`.

## Database and Generated Files

The local SQLite database contains:

- `students`: name, unique student ID, email, and creation time
- `attendance`: date, check-in time, check-out time, and status

Generated files include:

```text
attendance.db
faces/<Name>_<StudentID>.jpg
faces/face_encodings.pkl
identification_log_YYYYMMDD.txt
```

Back up `attendance.db` and the `faces/` directory if you need to preserve registered students and attendance history.

## Model Training

`trainer.ipynb` contains the training and evaluation workflow for the Siamese and classification models. It was designed for Google Colab and uses the CMU 11-785 HW2 Part 2 face dataset.

The dataset is intentionally not included in this repository because of its size. To retrain:

1. Download and extract the dataset separately.
2. Update the notebook paths to the extracted `classification_data` and verification-pair files.
3. Run the notebook with a GPU runtime when possible.
4. Export the final models to the filenames expected in `models/`.

The notebook's `.weights.h5` checkpoints are not directly interchangeable with the complete `.keras` model files used by the application. Rebuild the matching model architecture, load the checkpoint, and export the complete model before replacing an application model.

## Troubleshooting

### The camera does not open

- Check that a webcam is connected.
- Close other camera applications.
- Confirm that the operating system has granted camera permission to Python.
- If the camera is not device `0`, update the camera index in the application code.

### `dlib` or `face_recognition` fails to install

Install CMake and a C++ compiler first. On Windows, use Microsoft C++ Build Tools or a compatible prebuilt `dlib` wheel.

### OpenCV import errors

The requirements include both `opencv-python` and `opencv-contrib-python`. If they conflict, uninstall both and reinstall only `opencv-contrib-python`:

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python==4.8.1.78
```

### Models or face files are not found

Start the application from the project directory containing `main.py`. Model, face, and database paths are relative to the current working directory.

## Privacy

Face images, biometric encodings, and attendance records are sensitive personal data. Obtain consent before enrollment, restrict access to generated files, and follow the privacy and data-protection rules that apply in your location.

## License

This project is distributed under the [MIT License](LICENSE).
