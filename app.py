from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
import os
import subprocess  # For video conversion
from analysis import PostureAnalyzer  # Import your analysis class
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'  # Ensure processed videos are in "static"
app.secret_key = os.urandom(24)  # For session encryption (secure key)

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in files
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_FILE_DIR'] = './flask_session'  # Session storage directory
Session(app)  # Initialize Flask-Session

# Ensure upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handles video upload and processing."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)

        # Analyze the video
        analyzer = PostureAnalyzer()
        output_path, metrics_list = analyzer.analyze_video(video_path)
        average_metrics = analyzer.calculate_average_metrics(metrics_list)

        # Evaluate overall strain
        overall_strain = analyzer.evaluate_overall_strain(average_metrics)

        # Generate feedback based on strain prediction
        feedback = (
            "Great job! Your posture looks good. Keep maintaining a neutral neck and straight back."
            if overall_strain == "Not Straining"
            else "Your posture shows signs of strain. Try adjusting your seating position and taking breaks."
        )

        # Save processed video with a "_processed" suffix
        processed_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

        # Convert the processed video using FFmpeg
        converted_video_path = os.path.join(app.config['PROCESSED_FOLDER'], "converted_" + processed_filename)
        ffmpeg_command = [
            "ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", "-acodec", "aac", "-strict", "experimental", converted_video_path
        ]

        try:
            subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"✅ Video successfully converted: {converted_video_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Video conversion failed: {e}")
            converted_video_path = processed_video_path  # Fallback to original processed video if conversion fails

        # Delete the intermediate processed video to avoid duplication
        if os.path.exists(output_path):
            os.remove(output_path)

        # Store result data in session
        session['result_data'] = {
            'output_video': f"processed/{os.path.basename(converted_video_path)}",
            'average_metrics': {
                'neck_angle': average_metrics.neck_angle,
                'back_angle': average_metrics.back_angle,
                'shoulder_symmetry': average_metrics.shoulder_symmetry,
                'hip_alignment': average_metrics.hip_alignment,
                'hip_deviation_angle': average_metrics.hip_deviation_angle,
                'sentiment': average_metrics.sentiment
            },
            'overall_strain': overall_strain,
            'feedback': feedback
        }

        return redirect(url_for('result'))

    return render_template('upload.html')


@app.route('/result')
def result():
    """Displays the result page with processed video and analysis."""
    if 'result_data' not in session:
        return redirect(url_for('upload'))  # Redirect if no result data is found

    result_data = session['result_data']
    print(f"Processed video available at: {result_data['output_video']}")  # Debugging log
    return render_template('result.html', result=result_data)


@app.route('/clear_results')
def clear_results():
    """Clears session result data and redirects to the landing page."""
    session.pop('result_data', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
