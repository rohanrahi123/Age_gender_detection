# Age and Gender Detection using OpenVINO

This project performs **real-time face detection**, **age prediction**, and **gender classification** using a webcam.  
It uses official **OpenVINO pre-trained models** for high-speed CPU inference.

## 🚀 Features

✅ Real-time webcam detection  
✅ Detect multiple faces  
✅ Predict Age  
✅ Predict Gender  
✅ Draw bounding box + label on screen  
✅ Uses Intel OpenVINO optimized models

---

## 📂 Project Structure

```bash
Age-Gender-Detection/
│── main.py
│── face-detection-retail-0005.xml
│── face-detection-retail-0005.bin
│── age-gender-recognition-retail-0013.xml
│── age-gender-recognition-retail-0013.bin
│── README.md

⚙️ Requirements

Install dependencies:

pip install openvino opencv-python numpy

🧠 Models Used
1️⃣ Face Detection Model
Model Name: face-detection-retail-0005
Source: Official OpenVINO Model Zoo
2️⃣ Age + Gender Model
Model Name: age-gender-recognition-retail-0013
Source: Official OpenVINO Model Zoo

📄 face-detection-retail-0005.xml 

Model architecture file for face detection.

## Purpose

Detect human faces in image/video frames.

## Format

OpenVINO IR format (.xml)

## Input Size

300 x 300

## Output

Bounding box coordinates + confidence score

📄 face-detection-retail-0005.bin

Weights file for face detection model.

## Purpose

Stores trained parameters used by XML architecture file.

## Required With

face-detection-retail-0005.xml


📄 age-gender-recognition-retail-0013.xml 

Model architecture for age and gender recognition.

## Purpose

Predicts:

- Age
- Gender

## Input Size

62 x 62 face image

## Output

- Estimated Age
- Male / Female Probability
📄 age-gender-recognition-retail-0013.bin 
# age-gender-recognition-retail-0013.bin

Weights file for age-gender model.

## Purpose

Stores trained parameters for age and gender prediction.

## Required With

age-gender-recognition-retail-0013.xml
