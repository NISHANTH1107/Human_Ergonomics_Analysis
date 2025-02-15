# 🏋️ Work-from-Home Ergonomics Assistant

This is a Flask-based web application that analyzes posture and sentiment from video uploads.  
It uses **MediaPipe** for pose detection, **FER (Facial Expression Recognition)** for sentiment analysis,  
and **FFmpeg** for video processing.

## 🚀 Features
- 📹 **Upload video** for posture & sentiment analysis  
- 🔍 **Analyzes posture metrics** (neck angle, back angle, symmetry, etc.)  
- 😊 **Detects facial sentiment**  
- 🎬 **Processes & displays analyzed video**  
- 📊 **Generates feedback on ergonomic improvements**  

---

## 🌐 Live Demo
Try out the application here : [example.com](example.com)

## 🛠️ Installation & Setup

### 1️⃣ Install Python (if not installed)
Ensure you have Python **3.8+** installed.  
Check with:
```sh
python --version
```

## 2️⃣ Clone This Repository
```sh
git clone https://github.com/Human_Ergonomics_Analysis.git
cd Human_Ergonomics_Analysis
```

## 3️⃣ Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
## 4️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
## 5️⃣ Ensure FFmpeg is Installed
```sh
ffmpeg -version
```
## 6️⃣ Run the Flask App
```sh
python app.py
```
🔹 Open your browser and go to your [localhost](http://127.0.0.1:5000/)

## 🎯 How It Works
1. Upload a video (ensure clear upper-body visibility).
2. The app analyzes posture and sentiment.
3. The processed video is displayed with feedback.

---

## 🛑 Troubleshooting

**FFmpeg Not Found Error?**
- Ensure FFmpeg is installed & added to System PATH.
- Try running `ffmpeg -version` in the terminal.
- If using VS Code, restart it after updating PATH.

**Flask Session Not Storing Data?**
- Delete the `flask_session/` folder and restart the app.

---

## 📜 License
This project is open-source and free to use under the MIT License.
