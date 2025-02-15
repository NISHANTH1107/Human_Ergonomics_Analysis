import mediapipe as mp
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from fer import FER  # Import facial expression recognition
from moviepy.editor import VideoFileClip

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
    sentiment: str  # Add sentiment analysis result

class PostureAnalyzer:
    def __init__(self, model_complexity: int = 2):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=0.5
        )
        self.detector = FER(mtcnn=True)  # Ensure FER is initialized with MTCNN
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
        
    def calculate_angle(self, point1, point2, point3) -> float:
        """Calculate angle between three points in degrees."""
        p1 = np.array([point1.x, point1.y])
        p2 = np.array([point2.x, point2.y])
        p3 = np.array([point3.x, point3.y])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle
    
    def calculate_vertical_deviation(self, point1, point2) -> float:
        """Calculate deviation from vertical line in degrees."""
        dx = point2.x - point1.x
        dy = point2.y - point1.y
        # Calculate angle from vertical (90 degrees)
        angle = abs(90 - abs(np.degrees(np.arctan2(dy, dx))))
        return angle

    def calculate_hip_deviation(self, l_hip, r_hip) -> float:
        """Calculate hip deviation from horizontal."""
        dx = r_hip.x - l_hip.x
        dy = r_hip.y - l_hip.y
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
        try:
            emotions = self.detector.detect_emotions(image)
            if emotions:
                emotion, score = self.detector.top_emotion(image)
                return emotion if emotion else "Neutral"
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error during sentiment analysis: {e}")
            return "Neutral"

    def analyze_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[PostureMetrics]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose detected in the image!")
            return None, None
        
        landmarks = results.pose_landmarks.landmark
        metrics = self._calculate_metrics(landmarks, image_rgb)
        annotated_image = self._draw_annotations(image.copy(), results, metrics)
        
        return annotated_image, metrics

    def _calculate_metrics(self, landmarks, image_rgb) -> PostureMetrics:
        l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

        vertical_ref_point = type('Point', (), {'x': nose.x, 'y': nose.y - 1.0})
        mid_shoulder = type('Point', (), {'x': (l_shoulder.x + r_shoulder.x) / 2, 'y': (l_shoulder.y + r_shoulder.y) / 2})
        mid_hip = type('Point', (), {'x': (l_hip.x + r_hip.x) / 2, 'y': (l_hip.y + r_hip.y) / 2})
        
        neck_angle = self.calculate_angle(vertical_ref_point, nose, l_shoulder)
        back_angle = self.calculate_vertical_deviation(mid_hip, mid_shoulder)
        shoulder_symmetry = abs(l_shoulder.y - r_shoulder.y) * 100
        hip_alignment = abs(l_hip.y - r_hip.y) * 100
        hip_deviation_angle = self.calculate_hip_deviation(l_hip, r_hip)
        
        back_strain = self.assess_strain_level(back_angle, self.thresholds['back'], is_vertical=True)
        neck_strain = self.assess_strain_level(neck_angle, self.thresholds['neck'])
        sentiment = self.analyze_sentiment(image_rgb)

        return PostureMetrics(
        neck_angle=round(neck_angle, 4),
        back_angle=round(back_angle, 4),
        shoulder_symmetry=round(shoulder_symmetry, 4),
        hip_alignment=round(hip_alignment, 4),
        hip_deviation_angle=round(hip_deviation_angle, 4),
        back_bend_severity=back_strain,
        neck_strain_severity=neck_strain,
        sentiment=sentiment
    )

    def analyze_video(self, video_path: str) -> Tuple[str, List[PostureMetrics]]:
        clip = VideoFileClip(video_path)
        metrics_list = []
        output_frames = []

        frame_interval = 1 / 24  # Process at 24 frames per second
        for t in np.arange(0, clip.duration, frame_interval):
            frame = clip.get_frame(t)
            annotated_frame, metrics = self.analyze_image(frame)
            if annotated_frame is not None and metrics is not None:
                output_frames.append(annotated_frame)
                metrics_list.append(metrics)

        # Create output directory if it doesn’t exist
        output_dir = "static/processed"
        os.makedirs(output_dir, exist_ok=True)

        # Save output video
        filename = os.path.basename(video_path)
        output_filename = os.path.splitext(filename)[0] + "_processed.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        height, width, _ = output_frames[0].shape
        output_clip = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))

        for frame in output_frames:
            output_clip.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        output_clip.release()
        clip.close()

        return output_path, metrics_list


    def _draw_annotations(self, image: np.ndarray, results, metrics: PostureMetrics) -> np.ndarray:
        """Draw pose landmarks and add posture information to the image."""
        # Draw skeleton
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
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
        cv2.line(image, (mid_hip[0], int(h)), (mid_hip[0], 0), (0, 255, 0), 2)
        
        # Draw actual back line (red if severe, yellow if moderate/mild)
        line_color = (0, 0, 255) if metrics.back_bend_severity == StrainLevel.SEVERE else (0, 255, 255)
        cv2.line(image, mid_hip, mid_shoulder, line_color, 2)
        
        # Add metrics text with updated back angle interpretation
        font = cv2.FONT_HERSHEY_SIMPLEX
        metrics_text = [
            f"Back Deviation: {metrics.back_angle:.1f}° - {metrics.back_bend_severity.value}",
            f"Neck Angle: {metrics.neck_angle:.1f}° - {metrics.neck_strain_severity.value}",
            f"Hip Deviation: {metrics.hip_deviation_angle:.1f}°",
            f"Shoulder Symmetry: {metrics.shoulder_symmetry:.1f}%",
            f"Sentiment: {metrics.sentiment}"
        ]
        
        y_position = 30
        for text in metrics_text:
            color = (0, 0, 255) if "SEVERE" in text else (0, 0, 0)
            cv2.putText(image, text, (10, y_position), font, 0.7, color, 2)
            y_position += 30
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _generate_report(self, metrics: PostureMetrics) -> List[str]:
        """Generate a detailed posture analysis report."""
        report = ["Posture Analysis Report:"]
        
        if metrics.back_bend_severity != StrainLevel.NONE:
            report.append(f"⚠️ Back Deviation: {metrics.back_bend_severity.value} ({metrics.back_angle:.1f}° from vertical)")
            report.append("➡️ Recommendation: Align your back with the vertical axis and maintain upright posture")
        
        if metrics.neck_strain_severity != StrainLevel.NONE:
            report.append(f"⚠️ Neck Issue: {metrics.neck_strain_severity.value} ({metrics.neck_angle:.1f}°)")
            report.append("➡️ Recommendation: Adjust screen height and maintain neutral neck position")
        
        if metrics.hip_deviation_angle > self.thresholds['hip_angle']:
            report.append(f"⚠️ Hip Misalignment: {metrics.hip_deviation_angle:.1f}° deviation from horizontal")
            report.append("➡️ Recommendation: Level your hips and consider ergonomic seating")
        
        if len(report) == 1:
            report.append("✅ Good posture! All metrics within healthy ranges.")
        
        report.append(f"Sentiment: {metrics.sentiment}")
            
        return report
    
    def calculate_average_metrics(self, metrics_list: List[PostureMetrics]) -> Optional[PostureMetrics]:
        if not metrics_list:
            return None

        avg_neck_angle = round(sum(m.neck_angle for m in metrics_list) / len(metrics_list), 4)
        avg_back_angle = round(sum(m.back_angle for m in metrics_list) / len(metrics_list), 4)
        avg_shoulder_symmetry = round(sum(m.shoulder_symmetry for m in metrics_list) / len(metrics_list), 4)
        avg_hip_alignment = round(sum(m.hip_alignment for m in metrics_list) / len(metrics_list), 4)
        avg_hip_deviation_angle = round(sum(m.hip_deviation_angle for m in metrics_list) / len(metrics_list), 4)

        avg_back_bend_severity = self.assess_strain_level(avg_back_angle, self.thresholds['back'], is_vertical=True)
        avg_neck_strain_severity = self.assess_strain_level(avg_neck_angle, self.thresholds['neck'])

        sentiment_counts = {}
        for m in metrics_list:
            sentiment_counts[m.sentiment] = sentiment_counts.get(m.sentiment, 0) + 1
        most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)

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


if __name__ == "__main__":
    analyzer = PostureAnalyzer()
    video_path = "example_video.mp4"
    output_path, metrics_list = analyzer.analyze_video(video_path)
    print(f"Processed video saved to: {output_path}")
    
    # Calculate average metrics
    average_metrics = analyzer.calculate_average_metrics(metrics_list)
    print("Average Metrics:", average_metrics)
    
    # Evaluate overall strain
    if average_metrics:
        strain_status = analyzer.evaluate_overall_strain(average_metrics)
        print(f"Overall Strain Status: {strain_status}")
