from ultralytics import YOLO

if __name__ == "__main__":
    weights_path = r"weights/trial1/best.pt"
    model = YOLO(weights_path)
    model.predict(source=r"dataset/fromLigen/testvideo.mp4",  
                  save=True)