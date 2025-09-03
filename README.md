# emotion-detection
# 🎭 Emotion Detection System

An AI-powered **real-time emotion detection system** that uses **Java, OpenCV, and ONNX Runtime** to recognize human emotions from live webcam feed or uploaded images.  
This project demonstrates how computer vision and deep learning can be integrated to detect emotions such as **happy, sad, angry, neutral, surprised, etc.**

---

## 🚀 Features
- Real-time face detection using **OpenCV**
- Emotion classification using a **pre-trained ONNX model**
- Support for multiple emotions (happy, sad, neutral, angry, surprised, etc.)
- Integration with **web interface** for easy access
- Lightweight and fast inference

---

## 🏗️ Tech Stack
- **Java** (Core logic & backend)
- **OpenCV** (Face detection & preprocessing)
- **ONNX Runtime** (Emotion recognition model inference)
- **HTML/CSS/JS** (Frontend for web interface)
- **Spring Boot** (to serve as backend if hosted as a web app)

---
emotion-detection/
│── src/ # Java source files
│ ├── EmotionDetector.java # Main emotion detection logic
│ ├── OpenCamera.java # Webcam capture
│ └── ...
│
│── models/
│ └── emotion_model.onnx # Pre-trained ONNX model
│
│── resources/
│ └── index.html # Web interface
│
│── lib/ # External libraries (OpenCV, ONNX Runtime JARs)
│
│── README.md # Project documentation




---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
javac -cp ".;lib/*" src/EmotionDetector.java
java -cp ".;lib/*;src" EmotionDetector

## 📂 Project Structure

