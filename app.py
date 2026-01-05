# app.py
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import sqlite3
import hashlib
import os
import json
from datetime import datetime, timedelta
import time
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import requests
from datetime import datetime
# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

DB_NAME = "ergonomics.db"
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Comment out the problematic import and use simple analyzer
# from analysis import PostureAnalyzer

# Define a simple analyzer class inline
class PostureAnalyzer:
    def __init__(self):
        pass
    
    def analyze_image(self, image):
        import random
        class Metrics:
            neck_angle = round(random.uniform(60, 90), 1)
            back_angle = round(random.uniform(5, 25), 1)
            shoulder_symmetry = round(random.uniform(0, 20), 1)
            hip_alignment = round(random.uniform(0, 15), 1)
            hip_deviation_angle = round(random.uniform(0, 25), 1)
            back_bend_severity = type('StrainLevel', (), {'value': 'No strain'})()
            neck_strain_severity = type('StrainLevel', (), {'value': 'No strain'})()
            sentiment = random.choice(["Happy", "Neutral", "Focused"])
        
        return image, Metrics()

analyzer = PostureAnalyzer()

# Initialize posture analyzer
analyzer = PostureAnalyzer()

# ---------------- DATABASE ----------------
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    db = get_db()
    
    # Users table
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Sessions table
    db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            total_minutes INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Posture data table
    db.execute("""
        CREATE TABLE IF NOT EXISTS posture_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            neck_angle REAL,
            back_angle REAL,
            shoulder_symmetry REAL,
            hip_alignment REAL,
            hip_deviation_angle REAL,
            sentiment TEXT,
            back_strain_level TEXT,
            neck_strain_level TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Daily statistics table
    db.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            total_minutes INTEGER DEFAULT 0,
            avg_neck_angle REAL,
            avg_back_angle REAL,
            avg_shoulder_symmetry REAL,
            avg_hip_alignment REAL,
            avg_hip_deviation_angle REAL,
            bad_posture_percent REAL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date)
        )
    """)
    
    db.commit()

init_db()

# ---------------- HELPER FUNCTIONS ----------------
def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return fn(*args, **kwargs)
    return wrapper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user_id():
    return session.get('user_id')

def save_posture_data(session_id, metrics):
    """Save posture metrics to database"""
    db = get_db()
    db.execute("""
        INSERT INTO posture_data 
        (session_id, neck_angle, back_angle, shoulder_symmetry, hip_alignment, 
         hip_deviation_angle, sentiment, back_strain_level, neck_strain_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        session_id,
        metrics.neck_angle,
        metrics.back_angle,
        metrics.shoulder_symmetry,
        metrics.hip_alignment,
        metrics.hip_deviation_angle,
        metrics.sentiment,
        metrics.back_bend_severity.value,
        metrics.neck_strain_severity.value
    ))
    db.commit()

def calculate_daily_stats(user_id, date_str):
    """Calculate and store daily statistics"""
    db = get_db()
    
    # Get all posture data for the day
    data = db.execute("""
        SELECT * FROM posture_data 
        WHERE session_id IN (
            SELECT id FROM sessions 
            WHERE user_id = ? AND DATE(start_time) = ?
        )
    """, (user_id, date_str)).fetchall()
    
    if not data:
        return None
    
    # Calculate averages
    total_records = len(data)
    avg_neck = sum(d['neck_angle'] for d in data) / total_records
    avg_back = sum(d['back_angle'] for d in data) / total_records
    avg_shoulder = sum(d['shoulder_symmetry'] for d in data) / total_records
    avg_hip_align = sum(d['hip_alignment'] for d in data) / total_records
    avg_hip_dev = sum(d['hip_deviation_angle'] for d in data) / total_records
    
    # Calculate bad posture percentage (based on thresholds)
    bad_count = 0
    for d in data:
        if d['back_angle'] > 15 or d['neck_angle'] < 60:  # Adjust thresholds as needed
            bad_count += 1
    bad_percent = (bad_count / total_records) * 100
    
    # Get total minutes for the day
    total_minutes = db.execute("""
        SELECT COALESCE(SUM(total_minutes), 0) as total 
        FROM sessions 
        WHERE user_id = ? AND DATE(start_time) = ?
    """, (user_id, date_str)).fetchone()['total']
    
    # Store or update daily stats
    db.execute("""
        INSERT OR REPLACE INTO daily_stats 
        (user_id, date, total_minutes, avg_neck_angle, avg_back_angle, 
         avg_shoulder_symmetry, avg_hip_alignment, avg_hip_deviation_angle, bad_posture_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, date_str, total_minutes, avg_neck, avg_back, avg_shoulder,
        avg_hip_align, avg_hip_dev, bad_percent
    ))
    db.commit()
    
    return {
        'date': date_str,
        'total_minutes': total_minutes,
        'bad_posture_percent': bad_percent,
        'avg_metrics': {
            'neck_angle': round(avg_neck, 1),
            'back_angle': round(avg_back, 1),
            'shoulder_symmetry': round(avg_shoulder, 1),
            'hip_alignment': round(avg_hip_align, 1),
            'hip_deviation_angle': round(avg_hip_dev, 1)
        }
    }

def get_weekly_stats(user_id):
    """Get statistics for the last 7 days"""
    db = get_db()
    
    # Get data for last 7 days
    seven_days_ago = (datetime.now() - timedelta(days=6)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    stats = db.execute("""
        SELECT date, total_minutes, bad_posture_percent 
        FROM daily_stats 
        WHERE user_id = ? AND date BETWEEN ? AND ?
        ORDER BY date
    """, (user_id, seven_days_ago, today)).fetchall()
    
    # Fill in missing days
    result = []
    current_date = datetime.strptime(seven_days_ago, '%Y-%m-%d')
    end_date = datetime.strptime(today, '%Y-%m-%d')
    
    stat_dict = {s['date']: s for s in stats}
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        if date_str in stat_dict:
            result.append({
                'd': current_date.strftime('%b %d'),
                'bad_pct': round(stat_dict[date_str]['bad_posture_percent'], 1),
                'total_min': stat_dict[date_str]['total_minutes']
            })
        else:
            result.append({
                'd': current_date.strftime('%b %d'),
                'bad_pct': 0,
                'total_min': 0
            })
        current_date += timedelta(days=1)
    
    return result

def get_stress_areas(user_id):
    """Calculate stress areas for radar chart"""
    db = get_db()
    
    # Get today's data
    today = datetime.now().strftime('%Y-%m-%d')
    data = db.execute("""
        SELECT * FROM daily_stats 
        WHERE user_id = ? AND date = ?
    """, (user_id, today)).fetchone()
    
    if not data:
        # Return default values if no data
        return [
            {'area': 'Neck', 'stress': 30},
            {'area': 'Upper Spine', 'stress': 40},
            {'area': 'Lower Spine', 'stress': 50},
            {'area': 'Shoulders', 'stress': 35},
            {'area': 'Hip', 'stress': 45}
        ]
    
    # Calculate stress levels based on metrics (simplified)
    stress_levels = {
        'Neck': min(100, max(0, (data['avg_neck_angle'] - 50) * 3)),
        'Upper Spine': min(100, max(0, data['avg_back_angle'] * 5)),
        'Lower Spine': min(100, max(0, data['avg_back_angle'] * 4)),
        'Shoulders': min(100, max(0, data['avg_shoulder_symmetry'] * 2)),
        'Hip': min(100, max(0, data['avg_hip_deviation_angle'] * 3))
    }
    
    return [{'area': area, 'stress': round(level, 1)} for area, level in stress_levels.items()]

# ---------------- ROUTES ----------------
@app.route('/')
@login_required
def index():
    """Home page with live camera and session controls"""
    return render_template('index.html', username=session.get('username', 'User'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm = request.form.get('confirm_password')

        if not username or not password or not confirm:
            return render_template('register.html', error="All fields required")

        if password != confirm:
            return render_template('register.html', error="Passwords do not match")

        hashed = hash_password(password)

        try:
            db = get_db()
            db.execute(
                "INSERT INTO users (username, password) VALUES (?,?)",
                (username, hashed)
            )
            db.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error="Username already exists")

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        hashed = hash_password(password)

        db = get_db()
        user = db.execute(
            "SELECT id, username FROM users WHERE username=? AND password=?",
            (username, hashed)
        ).fetchone()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))

        return render_template('login.html', error="Invalid username or password")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/statistics')
@login_required
def statistics():
    """Statistics page with charts"""
    user_id = get_user_id()
    
    # Get weekly trend data
    weekly_stats = get_weekly_stats(user_id)
    
    # Get stress areas for radar chart
    stress_areas = get_stress_areas(user_id)
    
    # Calculate overall ergonomic score
    today_stats = next((s for s in weekly_stats if s['d'] == datetime.now().strftime('%b %d')), None)
    if today_stats and today_stats['bad_pct'] > 0:
        risk_level = "High Risk" if today_stats['bad_pct'] > 60 else "Medium Risk" if today_stats['bad_pct'] > 30 else "Low Risk"
    else:
        risk_level = "Low Risk"
    
    # Get total minutes for today
    today_minutes = today_stats['total_min'] if today_stats else 0
    
    return render_template('statistics.html', 
                         daily=weekly_stats,
                         areas=stress_areas,
                         risk_level=risk_level,
                         today_minutes=today_minutes)

@app.route('/ergo_chat')
@login_required
def ergo_chat():
    """Chatbot interface"""
    current_time = datetime.now()
    return render_template('chat.html', 
                         username=session.get('username', 'User'),
                         current_time=current_time)

@app.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    """Chatbot API endpoint"""
    try:
        message = request.json.get('message', '')
        
        # Simple response logic - in production, connect to local LLM
        responses = {
            'hello': 'Hello! How can I help you with your posture today?',
            'posture': 'Maintaining good posture involves keeping your back straight, shoulders relaxed, and screen at eye level.',
            'stretch': 'Try the cat-cow stretch: Arch your back upward, then downward, repeating 10 times.',
            'pain': 'If you\'re experiencing pain, take a break and do some gentle stretches. Consider consulting a doctor if pain persists.',
            'default': 'I\'m here to help with ergonomic advice. You can ask me about posture, stretches, or pain relief.'
        }
        
        msg_lower = message.lower()
        reply = responses.get('default')
        
        for key in responses:
            if key in msg_lower and key != 'default':
                reply = responses[key]
                break
        
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'reply': f'Error: {str(e)}'}), 500

@app.route('/api/start_session', methods=['POST'])
@login_required
def start_session():
    """Start a new session"""
    try:
        user_id = get_user_id()
        db = get_db()
        
        # End any existing active session
        db.execute("""
            UPDATE sessions SET end_time = CURRENT_TIMESTAMP 
            WHERE user_id = ? AND end_time IS NULL
        """, (user_id,))
        
        # Start new session
        cursor = db.execute("""
            INSERT INTO sessions (user_id) VALUES (?)
        """, (user_id,))
        session_id = cursor.lastrowid
        db.commit()
        
        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/end_session', methods=['POST'])
@login_required
def end_session():
    """End current session"""
    try:
        user_id = get_user_id()
        session_id = request.json.get('session_id')
        minutes = request.json.get('minutes', 0)
        
        db = get_db()
        db.execute("""
            UPDATE sessions 
            SET end_time = CURRENT_TIMESTAMP, total_minutes = ?
            WHERE id = ? AND user_id = ?
        """, (minutes, session_id, user_id))
        db.commit()
        
        # Calculate daily stats
        calculate_daily_stats(user_id, datetime.now().strftime('%Y-%m-%d'))
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze_frame', methods=['POST'])
@login_required
def analyze_frame():
    """Analyze a single frame from webcam"""
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400
        
        file = request.files['frame']
        
        # Read image
        nparr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Analyze posture
        annotated_image, metrics = analyzer.analyze_image(image)
        
        if metrics:
            # Save metrics if there's an active session
            session_id = request.args.get('session_id')
            if session_id:
                save_posture_data(session_id, metrics)
            
            return jsonify({
                'success': True,
                'neck_angle': round(metrics.neck_angle, 1),
                'back_angle': round(metrics.back_angle, 1),
                'shoulder_symmetry': round(metrics.shoulder_symmetry, 1),
                'hip_alignment': round(metrics.hip_alignment, 1),
                'hip_deviation_angle': round(metrics.hip_deviation_angle, 1),
                'sentiment': metrics.sentiment,
                'back_strain': metrics.back_bend_severity.value,
                'neck_strain': metrics.neck_strain_severity.value,
                'status': 'Good Posture' if metrics.back_bend_severity.value == 'No strain' and metrics.neck_strain_severity.value == 'No strain' else 'Improve Posture'
            })
        else:
            return jsonify({
                'success': False,
                'status': 'No pose detected',
                'neck_angle': 0,
                'back_angle': 0,
                'shoulder_symmetry': 0,
                'hip_alignment': 0,
                'hip_deviation_angle': 0
            })
            
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat_with_llm', methods=['POST'])
@login_required
def chat_with_llm():
    """Chat with local Llama 3.2 model"""
    try:
        data = request.json
        message = data.get('message', '')
        
        # Ergonomic system prompt
        system_prompt = """You are ErgoAssist, a helpful ergonomic assistant. You provide expert advice on:
        1. Posture correction and maintenance
        2. Ergonomic workstation setup
        3. Stretch and exercise routines for office workers
        4. Pain relief and prevention techniques
        5. Workplace wellness and productivity tips

        Be specific, practical, and encouraging in your responses. Provide actionable advice.
        If asked about medical conditions, recommend consulting a healthcare professional.
        Keep responses concise and focused on ergonomic principles.
        """
        
        # Prepare the prompt for Llama
        prompt = f"""{system_prompt}

        User: {message}

        Assistant:"""
        
        # Try to connect to local Llama 3.2
        try:
            # Option 1: Using Ollama (if you have it installed)
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.2',  # or whatever your model name is
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'max_tokens': 500
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get('response', '').strip()
            else:
                # If Ollama fails, try direct llama.cpp
                reply = generate_with_llama_cpp(prompt)
                
        except Exception as e:
            print(f"Error connecting to local LLM: {e}")
            # Fallback to rule-based responses
            reply = get_ergonomic_fallback_response(message)
        
        return jsonify({
            'success': True,
            'reply': reply,
            'model': 'llama3.2'
        })
        
    except Exception as e:
        print(f"Error in chat_with_llm: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'reply': 'I apologize, but I encountered an error. Please try again.'
        }), 500

def generate_with_llama_cpp(prompt):
    """Try to use llama.cpp directly"""
    try:
        # Path to your llama.cpp executable and model
        # Update these paths to match your setup
        llama_cpp_path = "C:/path/to/llama.cpp/main.exe"  # Update this
        model_path = "C:/path/to/models/llama-3.2.gguf"  # Update this
        
        # Prepare the command
        cmd = [
            llama_cpp_path,
            '-m', model_path,
            '-p', prompt,
            '-n', '500',  # max tokens
            '--temp', '0.7',
            '--repeat_penalty', '1.1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout.strip()
        
    except Exception as e:
        print(f"Error with llama.cpp: {e}")
        return get_ergonomic_fallback_response(prompt)

def get_ergonomic_fallback_response(message):
    """Fallback responses when LLM is not available"""
    msg_lower = message.lower()
    
    # Posture related
    if any(word in msg_lower for word in ['posture', 'sit', 'standing', 'slouch']):
        return """To improve your sitting posture:
        1. Keep feet flat on the floor
        2. Knees at 90 degrees, slightly lower than hips
        3. Back supported with lumbar support
        4. Shoulders relaxed, not hunched
        5. Screen at eye level, about arm's length away
        6. Take micro-breaks every 30 minutes
        
        Try the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds."""
    
    # Stretch related
    elif any(word in msg_lower for word in ['stretch', 'pain', 'ache', 'tight']):
        return """For immediate relief, try these stretches:
        
        **Neck Relief:**
        1. Chin tucks: Gently pull chin straight back
        2. Neck rotations: Slowly turn head side to side
        
        **Shoulder Relief:**
        1. Doorway stretch: Place forearms on doorframe, lean forward
        2. Shoulder rolls: Roll shoulders backward 10 times
        
        **Back Relief:**
        1. Cat-cow: On hands and knees, alternate arching and rounding back
        2. Child's pose: Sit back on heels, reach arms forward
        
        Hold each stretch for 15-30 seconds, repeat 2-3 times."""
    
    # Ergonomics setup
    elif any(word in msg_lower for word in ['setup', 'desk', 'chair', 'monitor', 'keyboard']):
        return """Optimal ergonomic setup:
        
        **Chair:**
        - Adjust height so feet are flat
        - Use lumbar support
        - Armrests at elbow height
        
        **Desk:**
        - Height allows 90-degree elbow angle
        - Keep frequently used items within reach
        
        **Monitor:**
        - Top of screen at eye level
        - 20-40 inches from eyes
        - Tilt slightly backward (10-20 degrees)
        
        **Keyboard & Mouse:**
        - Keyboard at elbow height
        - Mouse close to keyboard
        - Wrists straight, not bent"""
    
    # Break reminders
    elif any(word in msg_lower for word in ['break', 'rest', 'pause', 'interval']):
        return """Recommended break schedule:
        
        **Every 30 minutes:** Micro-break (30-60 seconds)
        - Stand up, stretch, change focus
        
        **Every 60-90 minutes:** Short break (5-10 minutes)
        - Walk around, hydrate, rest eyes
        
        **Every 3-4 hours:** Longer break (15-30 minutes)
        - Eat, short walk, relaxation
        
        Use Pomodoro technique: 25 minutes work, 5 minutes break."""
    
    # General wellness
    elif any(word in msg_lower for word in ['wellness', 'health', 'exercise', 'move']):
        return """Workplace wellness tips:
        
        1. Stay hydrated (8 glasses daily)
        2. Practice deep breathing to reduce stress
        3. Incorporate movement into your day
        4. Use standing desk if available
        5. Practice good sleep hygiene
        6. Manage stress with mindfulness
        
        Remember: Small, consistent changes make the biggest difference!"""
    
    # Default response
    else:
        return """I'm here to help with ergonomic advice! You can ask me about:
        
        • Sitting and standing posture
        • Desk and chair setup
        • Stretches for neck, back, and shoulders
        • Break schedules and movement tips
        • Workplace wellness strategies
        • Pain prevention techniques
        
        What would you like to know about today?"""

@app.route('/api/session_stats')
@login_required
def session_stats():
    """Get current session statistics"""
    try:
        user_id = get_user_id()
        db = get_db()
        
        # Get current session data
        session_data = db.execute("""
            SELECT id, start_time, total_minutes 
            FROM sessions 
            WHERE user_id = ? AND end_time IS NULL 
            ORDER BY start_time DESC LIMIT 1
        """, (user_id,)).fetchone()
        
        if session_data:
            # Get posture data for this session
            posture_data = db.execute("""
                SELECT COUNT(*) as count, 
                       AVG(neck_angle) as avg_neck,
                       AVG(back_angle) as avg_back,
                       AVG(shoulder_symmetry) as avg_shoulder
                FROM posture_data 
                WHERE session_id = ?
            """, (session_data['id'],)).fetchone()
            
            return jsonify({
                'session_id': session_data['id'],
                'duration': session_data['total_minutes'],
                'frames_analyzed': posture_data['count'] if posture_data else 0,
                'avg_neck_angle': round(posture_data['avg_neck'], 1) if posture_data and posture_data['avg_neck'] else 0,
                'avg_back_angle': round(posture_data['avg_back'], 1) if posture_data and posture_data['avg_back'] else 0
            })
        else:
            return jsonify({'no_session': True})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)