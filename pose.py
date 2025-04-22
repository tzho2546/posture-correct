from ultralytics import YOLO

# Path to the dataset configuration file
data_yaml = '/home/user/posture-correct/custom_data.yaml'

# Initialize a YOLOv8 model (you can choose a pre-trained model like 'yolov8n', 'yolov8s', etc.)
model = YOLO('best.pt')  # You can choose another version like yolov8s.pt for better performance

# Train the model
model.train(data=data_yaml, epochs=64, batch=16, imgsz=640)
