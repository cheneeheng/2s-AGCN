#! /bin/sh

IMAGE_NAME="2s-agcn:cuda11.1.1-cudnn8-devel-ubuntu20.04"

DATA_PATH="/home/dhm/workspace/deployment/data/2s-AGCN"
CODE_PATH="/home/dhm/workspace/deployment/code/2s-AGCN"

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --shm-size 8g --runtime=nvidia \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    --mount type=bind,source=${CODE_PATH},target=/code/openpose \
    --mount type=bind,source=${DATA_PATH},target=/data/openpose \
    ${TARGET_TAG} $1
