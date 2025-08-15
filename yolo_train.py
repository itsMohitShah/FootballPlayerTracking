from ultralytics import YOLO
from ultralytics import settings

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    dataset_yaml = r"dataset\roboflow\football-players-detection.v19-yolo11m.yolov11\data.yaml"
    settings.update({"runs_dir": "runs",
                     "datasets_dir": "dataset",
                     "weights_dir": "weights"})
    results = model.train(data = dataset_yaml,
                        epochs = 100,
                        imgsz = 640,
                        workers = 1)


