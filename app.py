from flask import Flask, render_template, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
from queue import Queue
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)
camera1 = cv2.VideoCapture("video.mp4")  # Change this to your video source

# Load the detection and segmentation models
detection_model = YOLO("yolov8n.pt")  # YOLOv8 object detection
segmentation_model = YOLO("yolov8n-seg.pt")  # YOLOv8 segmentation

q1 = Queue()

def process_camera(camera, q):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            height, width, _ = frame.shape

            # Step 1: Perform object detection for crowd counting
            detection_results = detection_model.predict(frame, device="mps")
            detection_result = detection_results[0]

            bboxes = np.array(detection_result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(detection_result.boxes.cls.cpu(), dtype="int")

            # Step 2: Perform segmentation using YOLOv8-seg
            segmentation_results = segmentation_model.predict(frame, device="mps")
            segmentation_result = segmentation_results[0]

            segmentation_masks = segmentation_result.masks.data.cpu().numpy()
            seg_classes = np.array(segmentation_result.boxes.cls.cpu(), dtype="int")

            # Filter for only 'person' class (class id 0) in both detection and segmentation
            person_mask = classes == 0
            person_seg_mask = seg_classes == 0

            bboxes = bboxes[person_mask]
            segmentation_masks = segmentation_masks[person_seg_mask]

            # Grid size (adjust as needed)
            grid_size = 4
            grid_height = height // grid_size
            grid_width = width // grid_size

            # Crowd count per grid
            crowd_grid = np.zeros((grid_size, grid_size))

            # Step 3: Count people in each grid cell for crowd density
            for bbox in bboxes:
                (x, y, x2, y2) = bbox
                cx = (x + x2) // 2  # Center of the bounding box
                cy = (y + y2) // 2

                # Determine which grid cell the person is in
                grid_x = min(cx // grid_width, grid_size - 1)
                grid_y = min(cy // grid_height, grid_size - 1)

                crowd_grid[grid_y, grid_x] += 1

            # Step 4: Draw the grid and apply color based on density
            for i in range(grid_size):
                for j in range(grid_size):
                    # Coordinates of the grid cell
                    x1 = j * grid_width
                    y1 = i * grid_height
                    x2 = (j + 1) * grid_width
                    y2 = (i + 1) * grid_height

                    # Color the grid based on the number of people
                    count = crowd_grid[i, j]

                    if count >= 6:  # High density
                        color = (0, 0, 255)  # Red
                    elif 3 <= count < 6:  # Moderate density
                        color = (0, 255, 255)  # Yellow
                    else:  # Low density
                        color = (0, 255, 0)  # Green

                    # Draw a rectangle for each grid cell with the appropriate color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Add text to display the crowd count in each cell
                    cv2.putText(frame, str(int(count)), (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Step 5: Draw bounding boxes for people and apply segmentation masks
            for bbox, mask in zip(bboxes, segmentation_masks):
                (x, y, x2, y2) = bbox
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                # Resize mask to fit the frame
                mask = cv2.resize(mask, (width, height))

                # Create a colored mask
                colored_mask = np.zeros_like(frame)
                if len(bboxes) > 2:  # If there are more than 2 people, use yellow for segmentation
                    colored_mask[mask > 0.5] = [0, 255, 255]  # Yellow mask for groups of people
                else:
                    colored_mask[mask > 0.5] = [0, 255, 0]  # Green mask for individuals

                # Overlay the segmentation mask on the frame
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

            # Step 6: Convert the frame to bytes and yield it with the crowd count
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Calculate total crowd count and put it in the queue
            total_crowd_count = len(bboxes)
            q.put(total_crowd_count)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video1')
def video1():
    return Response(process_camera(camera1, q1), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_crowd_count():
    while True:
        # Get crowd counts from the queue
        crowd_count = q1.get()
        yield 'data: %s\n\n' % str(crowd_count)

@app.route('/crowd-count')
def crowd_count1():
    return Response(generate_crowd_count(), mimetype='text/event-stream')

if __name__ == "__main__":
    # Start the thread for camera processing
    t1 = threading.Thread(target=process_camera, args=(camera1, q1))
    t1.daemon = True
    t1.start()

    # Start the thread for crowd count generation
    t = threading.Thread(target=generate_crowd_count)
    t.daemon = True
    t.start()

    # Get the port from the environment (default is 8001)
    port = int(os.environ.get("PORT", 8001))
    
    # Start the Flask app on host 0.0.0.0 to allow external access
    app.run(debug=True, port=port, host="0.0.0.0")
