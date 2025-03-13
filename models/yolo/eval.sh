yolo classify \
    val \
    model=/repo/models/yolo/trains/train2/weights/best.pt \
    data=/repo/data/testing/dataset.yaml\
    imgsz=224 \
    project=evals/ \
    name=eval-train2