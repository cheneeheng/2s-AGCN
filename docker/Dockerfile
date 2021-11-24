# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG UID_ARG=1000

RUN apt-get update -y
RUN apt-get install git -y
RUN apt-get install sudo -y

ENV TZ=Europe/Berlin
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN apt-get install ffmpeg libsm6 libxext6  -y

# Python 3.8
RUN apt-get install -y python3.8 python3.8-dev
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN apt-get install -y python3-pip
## Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
## Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install pip requirements
WORKDIR /code
COPY docker/requirements.txt /code/
RUN python -m pip install -r requirements.txt

RUN apt-get install python3-tk -y

# python name link.
RUN ln -s /usr/bin/python3 /usr/bin/python3-chen

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u $UID_ARG --disabled-password --gecos "" chen
RUN echo "chen:chen" | chpasswd 
RUN usermod -aG sudo chen
USER chen
WORKDIR /home/chen/2s-AGCN

EXPOSE 5432

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# File wsgi.py was not found in subfolder: 'django_webpage_form'. Please enter the Python path to wsgi file.
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "pythonPath.to.wsgi"]
CMD [ "bash" ]
