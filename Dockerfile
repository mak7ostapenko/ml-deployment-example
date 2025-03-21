FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Set the timezone
ENV TZ=Europe/Berlin
ARG DEBIAN_FRONTEND=noninteractive

# Update the system timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV APP_WORKDIR /home/depth_estimation
# Set the working directory inside the container
WORKDIR $APP_WORKDIR

RUN apt update -y && apt upgrade -y 
RUN apt-get install libopenblas-dev liblapack-dev -y
# needed for cv2 
RUN apt-get install ffmpeg libsm6 libxext6 -y

# Install Python dependencies
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade setuptools pip wheel
RUN pip --no-cache-dir install -r requirements_app.txt

EXPOSE 80

# CMD ["/bin/bash"]
CMD ["/bin/bash", "run_service.sh"]