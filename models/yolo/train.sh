yolo classify \
    train \
    name=train1 \
    project=./trains \
    data=../repo/data/training \
    model=yolo11m-cls.pt \
    epochs=20 \
    imgsz=224 \

    device=0