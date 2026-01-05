# analysis.py
import cv2
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from moviepy.editor import VideoFileClip

# Try to import MediaPipe with proper error handling
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not installed. Please install with: pip install mediapipe")

# Try to import FER with proper error handling
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("FER not installed. Please install with: pip install fer")

class StrainLevel(Enum):
    NONE = "No strain"
    MILD = "Mild strain risk"
    MODERATE = "Moderate strain risk"
    SEVERE = "Severe strain risk"

@dataclass
class PostureMetrics:
    neck_angle: float
    back_angle: float
    shoulder_symmetry: float
    hip_alignment: float
    hip_deviation_angle: float
    back_bend_severity: StrainLevel
    neck_strain_severity: StrainLevel
    sentiment: str

class PostureAnalyzer:
    def __init__(self, model_complexity: int = 1):  # Reduced complexity for better performance
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required but not installed.")
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Changed to False for video processing
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize FER only if available
        self.detector = FER(mtcnn=True) if FER_AVAILABLE else None
        
        self.thresholds = {
            'neck': {
                'optimal': 80,
                'mild': 70,
                'moderate': 60,
                'severe': 50
            },
            'back': {
                'optimal': 5,
                'mild': 10,
                'moderate': 15,
                'severe': 20
            },
            'shoulder': 15,
            'hip': 10,
            'hip_angle': 20
        }
        
        # For drawing landmarks
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def calculate_angle(self, point1, point2, point3) -> float:
        """Calculate angle between three points in degrees."""
        if None in [point1, point2, point3]:
            return 0.0
            
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def calculate_vertical_deviation(self, point1, point2) -> float:
        """Calculate deviation from vertical line in degrees."""
        if None in [point1, point2]:
            return 0.0
            
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        
        # Avoid division by zero
        if dx == 0:
            return 90.0
            
        angle_from_horizontal = np.degrees(np.arctan2(dy, dx))
        angle_from_vertical = abs(90 - abs(angle_from_horizontal))
        
        return angle_from_vertical

    def calculate_hip_deviation(self, l_hip, r_hip) -> float:
        """Calculate hip deviation from horizontal."""
        if None in [l_hip, r_hip]:
            return 0.0
            
        dx = r_hip.x - l_hip.x
        dy = r_hip.y - l_hip.y
        
        if dx == 0:
            return 90.0
            
        angle = np.degrees(np.arctan2(dy, dx))
        return abs(angle) % 90

    def assess_strain_level(self, angle: float, thresholds: Dict[str, float], is_vertical: bool = False) -> StrainLevel:
        """Determine strain level based on angle measurements."""
        if is_vertical:
            # For vertical measurements (like back angle), smaller angles are better
            if angle <= thresholds['optimal']:
                return StrainLevel.NONE
            elif angle <= thresholds['mild']:
                return StrainLevel.MILD
            elif angle <= thresholds['moderate']:
                return StrainLevel.MODERATE
            else:
                return StrainLevel.SEVERE
        else:
            # For other measurements (like neck angle), larger angles are better
            if angle >= thresholds['optimal']:
                return StrainLevel.NONE
            elif angle >= thresholds['mild']:
                return StrainLevel.MILD
            elif angle >= thresholds['moderate']:
                return StrainLevel.MODERATE
            else:
                return StrainLevel.SEVERE
            
    def analyze_sentiment(self, image: np.ndarray) -> str:
        """Analyze facial expression sentiment."""
        if self.detector is None:
            return "Neutral"
            
        try:
            # Convert to RGB for FER
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            emotions = self.detector.detect_emotions(image_rgb)
            
            if emotions:
                # Get the first face detected
                emotion_data = emotions[0]['emotions']
                # Find the emotion with highest score
                emotion = max(emotion_data, key=emotion_data.get)
                return emotion.capitalize()
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return "Neutral"

    def analyze_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[PostureMetrics]]:
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # To improve performance, optionally mark the image as not writeable to pass by reference
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if not results.pose_landmarks:
                print("No pose detected in the image!")
                return None, None
            
            landmarks = results.pose_landmarks.landmark
            metrics = self._calculate_metrics(landmarks, image_rgb)
            annotated_image = self._draw_annotations(image.copy(), results, metrics)
            
            return annotated_image, metrics
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None, None

    def _calculate_metrics(self, landmarks, image_rgb) -> PostureMetrics:
        try:
            l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

            # Create vertical reference point (above nose)
            vertical_ref_point = type('Point', (), {'x': nose.x, 'y': nose.y - 0.2})()  # 0.2 normalized units above nose
            
            # Calculate midpoints
            mid_shoulder_x = (l_shoulder.x + r_shoulder.x) / 2
            mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            mid_hip_x = (l_hip.x + r_hip.x) / 2
            mid_hip_y = (l_hip.y + r_hip.y) / 2
            
            mid_shoulder = type('Point', (), {'x': mid_shoulder_x, 'y': mid_shoulder_y})()
            mid_hip = type('Point', (), {'x': mid_hip_x, 'y': mid_hip_y})()
            
            # Calculate metrics
            neck_angle = self.calculate_angle(vertical_ref_point, nose, l_shoulder)
            back_angle = self.calculate_vertical_deviation(mid_hip, mid_shoulder)
            shoulder_symmetry = abs(l_shoulder.y - r_shoulder.y) * 100
            hip_alignment = abs(l_hip.y - r_hip.y) * 100
            hip_deviation_angle = self.calculate_hip_deviation(l_hip, r_hip)
            
            # Assess strain levels
            back_strain = self.assess_strain_level(back_angle, self.thresholds['back'], is_vertical=True)
            neck_strain = self.assess_strain_level(neck_angle, self.thresholds['neck'])
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            return PostureMetrics(
                neck_angle=round(neck_angle, 2),
                back_angle=round(back_angle, 2),
                shoulder_symmetry=round(shoulder_symmetry, 2),
                hip_alignment=round(hip_alignment, 2),
                hip_deviation_angle=round(hip_deviation_angle, 2),
                back_bend_severity=back_strain,
                neck_strain_severity=neck_strain,
                sentiment=sentiment
            )
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            # Return default metrics
            return PostureMetrics(
                neck_angle=0.0,
                back_angle=0.0,
                shoulder_symmetry=0.0,
                hip_alignment=0.0,
                hip_deviation_angle=0.0,
                back_bend_severity=StrainLevel.NONE,
                neck_strain_severity=StrainLevel.NONE,
                sentiment="Neutral"
            )

    def analyze_video(self, video_path: str) -> Tuple[str, List[PostureMetrics]]:
        """Analyze a video file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        clip = VideoFileClip(video_path)
        metrics_list = []
        output_frames = []

        # Process at 10 fps for better performance
        frame_interval = 1 / 10
        
        for t in np.arange(0, min(clip.duration, 30), frame_interval):  # Limit to 30 seconds
            try:
                frame = clip.get_frame(t)
                annotated_frame, metrics = self.analyze_image(frame)
                if annotated_frame is not None and metrics is not None:
                    output_frames.append(annotated_frame)
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"Error processing frame at {t}s: {e}")
                continue

        if not output_frames:
            raise ValueError("No frames could be processed from the video")

        # Create output directory
        output_dir = "static/processed"
        os.makedirs(output_dir, exist_ok=True)

        # Save output video
        filename = os.path.basename(video_path)
        output_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save frames as video
        height, width, _ = output_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
        
        for frame in output_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        clip.close()

        return output_path, metrics_list

    def _draw_annotations(self, image: np.ndarray, results, metrics: PostureMetrics) -> np.ndarray:
        """Draw pose landmarks and add posture information to the image."""
        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        h, w = image.shape[:2]
        landmarks = results.pose_landmarks.landmark
        
        # Get midpoints for shoulders and hips
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        mid_shoulder = (
            int((l_shoulder.x + r_shoulder.x) * w / 2),
            int((l_shoulder.y + r_shoulder.y) * h / 2)
        )
        mid_hip = (
            int((l_hip.x + r_hip.x) * w / 2),
            int((l_hip.y + r_hip.y) * h / 2)
        )
        
        # Draw vertical reference line from hips (green)
        cv2.line(image, (mid_hip[0], h), (mid_hip[0], 0), (0, 255, 0), 2)
        
        # Draw actual back line (red if severe, yellow if moderate/mild)
        line_color = (0, 0, 255) if metrics.back_bend_severity == StrainLevel.SEVERE else (0, 255, 255)
        cv2.line(image, mid_hip, mid_shoulder, line_color, 3)
        
        # Add metrics text
        font = cv2.FONT_HERSHEY_SIMPLEX
        metrics_text = [
            f"Back: {metrics.back_angle:.1f}° - {metrics.back_bend_severity.value}",
            f"Neck: {metrics.neck_angle:.1f}° - {metrics.neck_strain_severity.value}",
            f"Shoulder Sym: {metrics.shoulder_symmetry:.1f}%",
            f"Sentiment: {metrics.sentiment}"
        ]
        
        y_position = 30
        for text in metrics_text:
            color = (0, 0, 255) if "SEVERE" in text else (0, 255, 255) if "MILD" in text or "MODERATE" in text else (0, 255, 0)
            cv2.putText(image, text, (10, y_position), font, 0.6, color, 2)
            y_position += 30
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def calculate_average_metrics(self, metrics_list: List[PostureMetrics]) -> Optional[PostureMetrics]:
        if not metrics_list:
            return None

        avg_neck_angle = round(sum(m.neck_angle for m in metrics_list) / len(metrics_list), 2)
        avg_back_angle = round(sum(m.back_angle for m in metrics_list) / len(metrics_list), 2)
        avg_shoulder_symmetry = round(sum(m.shoulder_symmetry for m in metrics_list) / len(metrics_list), 2)
        avg_hip_alignment = round(sum(m.hip_alignment for m in metrics_list) / len(metrics_list), 2)
        avg_hip_deviation_angle = round(sum(m.hip_deviation_angle for m in metrics_list) / len(metrics_list), 2)

        avg_back_bend_severity = self.assess_strain_level(avg_back_angle, self.thresholds['back'], is_vertical=True)
        avg_neck_strain_severity = self.assess_strain_level(avg_neck_angle, self.thresholds['neck'])

        # Find most common sentiment
        sentiment_counts = {}
        for m in metrics_list:
            sentiment_counts[m.sentiment] = sentiment_counts.get(m.sentiment, 0) + 1
        most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "Neutral"

        return PostureMetrics(
            neck_angle=avg_neck_angle,
            back_angle=avg_back_angle,
            shoulder_symmetry=avg_shoulder_symmetry,
            hip_alignment=avg_hip_alignment,
            hip_deviation_angle=avg_hip_deviation_angle,
            back_bend_severity=avg_back_bend_severity,
            neck_strain_severity=avg_neck_strain_severity,
            sentiment=most_common_sentiment
        )
    
    def evaluate_overall_strain(self, metrics: PostureMetrics) -> str:
        """Evaluate overall strain based on neck and back metrics."""
        if any([
            metrics.back_bend_severity in [StrainLevel.MODERATE, StrainLevel.SEVERE],
            metrics.neck_strain_severity in [StrainLevel.MODERATE, StrainLevel.SEVERE]
        ]):
            return "Straining"
        return "Not Straining"


# Simple fallback analyzer for testing
class SimplePostureAnalyzer:
    """Simple analyzer for testing when MediaPipe is not available"""
    def __init__(self):
        self.thresholds = {
            'neck': {'optimal': 80, 'mild': 70, 'moderate': 60, 'severe': 50},
            'back': {'optimal': 5, 'mild': 10, 'moderate': 15, 'severe': 20}
        }
    
    def analyze_image(self, image: np.ndarray):
        # Return dummy metrics for testing
        import random
        metrics = PostureMetrics(
            neck_angle=round(random.uniform(50, 90), 2),
            back_angle=round(random.uniform(0, 30), 2),
            shoulder_symmetry=round(random.uniform(0, 20), 2),
            hip_alignment=round(random.uniform(0, 15), 2),
            hip_deviation_angle=round(random.uniform(0, 25), 2),
            back_bend_severity=StrainLevel.NONE,
            neck_strain_severity=StrainLevel.NONE,
            sentiment="Neutral"
        )
        return image, metrics

if __name__ == "__main__":
    try:
        analyzer = PostureAnalyzer()
        print("PostureAnalyzer initialized successfully")
    except ImportError as e:
        print(f"Using SimplePostureAnalyzer: {e}")
        analyzer = SimplePostureAnalyzer()