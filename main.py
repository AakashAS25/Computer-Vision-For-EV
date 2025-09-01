
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import kagglehub
import os


# ---------------------------
# Calibration (semi-automatic using lane markers)
# ---------------------------
def calibrate_lane_width(frame, lane_width_m=3.5):
    """
    Estimate pixel-to-meter ratio from lane markings (Hough line detection).
    If fails, default ratio is returned.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=200)

    if lines is not None:
        # Take first two vertical-ish lines (lane markers)
        verticals = [l[0] for l in lines if abs(l[0][2]-l[0][0]) < 30]  # near-vertical
        if len(verticals) >= 2:
            # pick two farthest lines
            x_coords = [l[0] for l in verticals]
            x_min, x_max = min(x_coords), max(x_coords)
            pixel_dist = abs(x_max - x_min)
            return lane_width_m / pixel_dist  # meters per pixel

    return 3.5 / 200  # default fallback ratio if no lanes detected





# ---------------------------
# Main Vehicle Detection + Tracking
# ---------------------------

def main(video_source=0):

    # Load YOLOv8 face detection model (model.pt from HuggingFace or yolov8n-face.pt)
    face_model_path = os.path.join(os.getcwd(), "model.pt")
    if not os.path.exists(face_model_path):
        print("YOLOv8 face model 'model.pt' not found! Please place it in the project directory.")
        face_model = None
    else:
        face_model = YOLO(face_model_path)

    # Download and load human detection model from KaggleHub
    human_model_path = kagglehub.model_download("thanh309/yolo11-human-detection/pyTorch/default")
    human_model_file = os.path.join(human_model_path, "human-detection.pt")
    human_model = YOLO(human_model_file)

    # Load YOLOv8 vehicle model (make sure yolov8n.pt is in your project directory)
    vehicle_model = YOLO("yolov8n.pt")

    # DeepSORT tracker
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calibration using first frame
    ret, frame = cap.read()
    if not ret:
        print("Camera not available.")
        return
    mpp = calibrate_lane_width(frame)
    print(f"Calibration: {mpp:.5f} meters per pixel")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = []



        # Detect vehicles
        vehicle_results = vehicle_model(frame, verbose=False)[0]
        for box in vehicle_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = vehicle_model.names[cls]
            if label in ["car", "truck", "bus", "motorbike"]:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        # Detect humans
        human_results = human_model(frame, verbose=False)[0]
        for box in human_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = human_model.names[cls] if hasattr(human_model, 'names') else "person"
            if label == "person" or label == 0 or label == "human":
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # Detect faces using YOLOv8 face model
        if face_model is not None:
            face_results = face_model(frame, verbose=False)[0]
            for box in face_results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                # Accept all detections as face
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "face"))

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for t in tracks:
            if not t.is_confirmed():
                continue
            track_id = t.track_id
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = t.get_det_class() if t.get_det_class() else "object"

            # Draw bounding box + label
            if label == "person":
                color = (0, 0, 255)  # Red for person
            elif label == "face":
                color = (255, 0, 255)  # Magenta for face
            else:
                color = (0, 255, 0)  # Green for vehicles
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id} {label}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cv2.imshow("Detection (Vehicle + Human)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0)  
