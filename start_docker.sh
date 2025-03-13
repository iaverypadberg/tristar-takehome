# Pull the ultralytics docker image down
docker pull ultralytics/ultralytics:latest

# run the docker image with path specified to the to the local data folder
sudo docker run --rm -v .:/repo  -it --ipc=host --gpus all --name tristar-takehome-container ultralytics/ultralytics:latest