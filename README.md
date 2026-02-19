
# AI-Based Driver Drowsiness Detection System

## Overview
This project implements a real-time AI-based driver drowsiness detection system using Python, OpenCV, and Machine Learning.

## Features
- Real-time face and eye detection
- Eye Aspect Ratio (EAR) based drowsiness detection
- Alarm alert system
- Modular project structure
- Automotive safety focused design

## Technologies Used
- Python
- OpenCV
- dlib
- NumPy
- Scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python main.py
```

## Project Structure

```
AI_Driver_Drowsiness_Detection/
│
├── main.py
├── requirements.txt
├── models/
├── utils/
│   ├── ear.py
│   └── alert.py
└── README.md
```

## Notes
Download shape_predictor_68_face_landmarks.dat from dlib's official site and place it inside the models/ directory.
