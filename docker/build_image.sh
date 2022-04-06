#! /bin/sh

if [ ${1} == 'CPU' ]; then
    echo "Building CPU version..."
    IMAGE_NAME="2s-agcn:ubuntu20.04"
    DOCKER_FILENAME = "Dockerfile.CPU"
elif [ ${1} == 'CU111' ]; then
    echo "Building cuda11.1.1 version..."
    IMAGE_NAME="2s-agcn:cuda11.1.1-cudnn8-devel-ubuntu20.04"
    DOCKER_FILENAME = "Dockerfile.CU111"
elif [ ${1} == 'CU113' ]; then
    echo "Building cuda11.3.1 version..."
    IMAGE_NAME="2s-agcn:cuda11.3.1-cudnn8-devel-ubuntu20.04"
    DOCKER_FILENAME = "Dockerfile.CU113"
else
    echo "Build failed, please specify the build type"
fi

echo "Building image : ${IMAGE_NAME}"
DOCKER_BUILDKIT=1 docker build \
    --file ${DOCKER_FILENAME} \
    --build-arg UNAME_ARG=$1 \
    --build-arg UID_ARG=$2 \
    --tag ${IMAGE_NAME} \
    .
echo "Built image : ${IMAGE_NAME}\n"
