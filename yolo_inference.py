from ultralytics import YOLO
import os
import wandb
from wandb.integration.ultralytics import add_wandb_callback
if __name__ == "__main__":
    with wandb.init(project="football-player-tracking", job_type="inference") as run:
        weights_path = r"weights\002_epochs100\best.pt"
        test_name = weights_path.split(os.sep)[-2]+"-"
        model = YOLO(weights_path)
        add_wandb_callback(model)
        model.predict(source=r"dataset/fromLigen/testvideo.mp4",  
                    name = test_name,
                    save=True, 
                    save_txt=True,
                    save_conf=True,
                    show_labels = False,
                    show_conf = False,
                    show = True)