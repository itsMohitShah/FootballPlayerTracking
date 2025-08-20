from ultralytics import YOLO

if __name__ == "__main__":
    weights = r"weights/002_epochs100/best.pt"
    source=r"dataset/fromLigen/testvideo.mp4"
    model = YOLO(weights)

    results = model.track(source=source,
                          name = "BotSortwithReID",
                          show=True,
                          show_labels=True,
                          show_conf=False,
                          save=True,
                          tracker = 'botsort.yaml',
                          line_width=1)