from ultralytics import YOLO

def load_model():
    model = YOLO("yolov8n.pt")
    model.to("cuda")
    return model

def detect(model, frame):
    results = model.track(frame, persist=True, verbose=False)
    return results