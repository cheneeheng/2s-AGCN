#! /bin/sh

IMAGE_NAME="2s-agcn:cuda11.1.1-cudnn8-devel-ubuntu20.04"

CODE_PATH="/home/dhm/workspace/demo_event/code/2s-AGCN"
DATA_PATH="/home/dhm/workspace/demo_event/data/2s-agcn"
OPENPOSE_DATA_PATH="/home/dhm/workspace/demo_event/data/openpose"

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --shm-size 8g --gpus=all \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    --mount type=bind,source=${CODE_PATH},target=/code/2s-AGCN \
    --mount type=bind,source=${DATA_PATH},target=/data/2s-agcn \
    --mount type=bind,source=${OPENPOSE_DATA_PATH},target=/data/openpose \
    ${IMAGE_NAME} $1
