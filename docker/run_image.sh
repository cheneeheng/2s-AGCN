#! /bin/sh

if [ ${1} = 'CPU' ]; then
    echo "Building CPU version..."
    IMAGE_NAME="cheneeheng/2s-agcn:ubuntu20.04"
elif [ ${1} = 'CU111' ]; then
    echo "Building cuda11.1.1 version..."
    IMAGE_NAME="cheneeheng/2s-agcn:cuda11.1.1-cudnn8-devel-ubuntu20.04"
elif [ ${1} = 'CU113' ]; then
    echo "Building cuda11.3.1 version..."
    IMAGE_NAME="cheneeheng/2s-agcn:cuda11.3.1-cudnn8-devel-ubuntu20.04"
else
    echo "Build failed, please specify the build type"
fi

# CODE_PATH="/home/dhm/workspace/demo_event/code/2s-AGCN"
# DATA_PATH="/home/dhm/workspace/demo_event/data/2s-AGCN"
# OPENPOSE_DATA_PATH="/home/dhm/workspace/demo_event/data/openpose"

CODE_PATH="/home/chen/work/12_AAGCN/2s-AGCN"
DATA_PATH="/home/chen/data/07_AAGCN"

# By using --device-cgroup-rule flag we grant the docker continer permissions -
# to the camera and usb endpoints of the machine.
# It also mounts the /dev directory of the host platform on the contianer
docker run -it --rm --shm-size 8g --gpus=all \
    --mount type=bind,source=/dev/shm,target=/dev/shm \
    --mount type=bind,source=${CODE_PATH},target=/code/2s-AGCN \
    --mount type=bind,source=${DATA_PATH},target=/code/2s-AGCN/data \
    ${IMAGE_NAME} ${2}
    # --mount type=bind,source=${OPENPOSE_DATA_PATH},target=/data/openpose \
