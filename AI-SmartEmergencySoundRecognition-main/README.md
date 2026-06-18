# AI-Enabled Sound Alert Wearable for the Hearing Impaired

## Project Overview

The AI-Enabled Sound Alert Wearable is an assistive technology project designed to help hearing-impaired individuals recognize important environmental sounds in real time. The system continuously monitors surrounding audio, classifies detected sounds using machine learning, and provides alerts through a vibration motor and an OLED display.

The project uses YAMNet for audio feature extraction and a Support Vector Machine (SVM) classifier for sound recognition. The final system is deployed on a Raspberry Pi, making it suitable for portable and real-time applications.

---

## Features

* Real-time environmental sound detection
* AI-powered sound classification
* Vibration-based alerts for users
* OLED display notifications
* Raspberry Pi deployment
* YAMNet-based feature extraction
* SVM-based sound classification

---

## Sound Classes

The model recognizes the following 13 sound categories:

* Car Horn
* Crying Baby
* Dog
* Door Knock
* Doorbell
* Fire/Smoke Alarm
* Glass Breaking
* Gunshot
* Rain
* Screaming
* Siren
* Thunderstorm
* Train

---

## System Workflow

```text
Microphone
    ↓
Audio Capture
    ↓
Audio Preprocessing
    ↓
YAMNet Feature Extraction
    ↓
SVM Classification
    ↓
Detected Sound
    ↓
OLED Display Alert + Vibration Alert
```

---

## Dataset

This project uses a combination of the ESC-50 environmental sound dataset and custom-collected audio samples.

### ESC-50 Classes

* car_horn
* crying_baby
* dog
* door_knock
* doorbell
* glass_breaking
* rain
* siren
* train

Dataset Source:
https://github.com/karolpiczak/ESC-50

### Custom Classes

* fire_smoke_alarm
* gunshot
* screaming
* thunderstorm

---

## Audio Preprocessing

The dataset underwent the following preprocessing steps:

1. Audio augmentation
2. Audio padding and trimming
3. Resampling to 16 kHz
4. Feature extraction using YAMNet embeddings
5. Dataset preparation for model training

---

## Machine Learning Pipeline

1. Collect and prepare audio samples
2. Apply preprocessing and augmentation
3. Extract embeddings using YAMNet
4. Train an SVM classifier
5. Evaluate model performance
6. Deploy the trained model on Raspberry Pi
7. Generate vibration and visual alerts for detected sounds

---

## Hardware Components

* Raspberry Pi
* USB Microphone
* SH1106 OLED Display
* Vibration Motor

---

## Technologies Used

* Python
* TensorFlow Lite
* YAMNet
* Scikit-learn
* NumPy
* Librosa
* SoundDevice
* Raspberry Pi GPIO
* OLED Display Libraries

---

## Future Improvements

* Mobile application integration
* Wearable device miniaturization
* Additional sound categories
* Wireless notification system
* Improved real-time performance
* Edge AI optimization

---

## Authors

Dhananjay A.V

---

## License

This project is licensed under the MIT License.
