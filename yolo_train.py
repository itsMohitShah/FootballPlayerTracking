from ultralytics import YOLO
from ultralytics import settings
import wandb
from wandb.integration.ultralytics import add_wandb_callback

if __name__ == "__main__":
    dataset_yaml = r"dataset\roboflow\football-players-detection.v19-yolo11m.yolov11\data.yaml"
    settings.update({"runs_dir": "runs",
                     "datasets_dir": "datasets",
                     "weights_dir": "weights"})
    with wandb.init(project="football-player-tracking", 
                    job_type="train") as run:
        model = YOLO("yolo11n.pt")
        add_wandb_callback(model,enable_model_checkpointing=True)
        results = model.train(project = "football-player-tracking",
                            data = dataset_yaml,
                            cfg = r'cfg\default.yaml',
                            epochs = 100,
                            imgsz = 640,
                            workers = 1)


