# Dockerfile that sets up a gym workspace
FROM ubuntu:16.04

MAINTAINER Lionel Blonde <lionel.blonde@unige.ch>

# Install ubuntu packages
# -q: produce output suitable for logging, omitting progress indicators
# DEBIAN_FRONTEND=noninteractive: anti-frontend (never interacts with you at all,
#     and makes the default answers be used for all questions
# -y: automatically answer 'yes' to prompts (less powerful than previous)
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    software-properties-common \
    libav-tools \
    python-dev \
    python3-dev \
    python-setuptools \
    libopenmpi-dev \
    openssh-client \
    openssh-server \
    python-pip \
    zlib1g-dev \
    python-opencv \
    libpq-dev \
    libjpeg-dev \
    curl \
    make \
    cmake \
    swig \
    freeglut3 \
    python-opengl \
    libboost-all-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libsdl2-2.0-0\
    libgles2-mesa-dev \
    libsdl2-dev \
    libosmesa6-dev \
    libglfw3-dev \
    libglew-dev \
    libgl1-mesa-glx \
    wget \
    unzip \
    git \
    python-gtkglext1 \
    ca-certificates \
    xpra \
    xserver-xorg-dev \
    virtualenv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download python and pip from the 'deadsnakes' ppa (more recent py versions packaged for Ubuntu)
# This is the simplest way to get Python 3.6 stuff via apt-get only (w/o the need for conda)
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3-pip
# Create a virtual environment with the 'deadsnakes' Python version 3.6
RUN virtualenv --python=python3.6 env
# Replace the default Python directories by the ones from the virtual environment via symlinks
# Note that it is in effect equivalent to activating the virtual environment
RUN rm /usr/bin/python
RUN rm /usr/bin/pip
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest
# Note that the pytest directory will only be created when after pytest is installed via pip

# Download openai custom-made patchelf version (line found in 'mujoco-py' Dockerfile)
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# Create environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV TERM xterm-256color

# Install pip packages
RUN pip install --upgrade pip
RUN pip install \
    pytest \
    pytest-instafail \
    PyYAML \
    tqdm \
    numpy \
    scipy \
    sklearn \
    pandas \
    Pillow \
    plotly \
    mpi4py \
    Cython \
    PyOpenGL \
    six \
    opencv-python \
    wrapt \
    && rm -rf ~/.cache/pip

# Install pip packages that have gpu implementations
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
RUN pip install \
    tensorflow-probability \
    && rm -rf ~/.cache/pip

# Install MuJoCo 1.50
ENV MUJOCO_PATH=/root/.mujoco
ENV LD_LIBRARY_PATH=${MUJOCO_PATH}/mjpro150/bin:${LD_LIBRARY_PATH}
RUN mkdir -p ${MUJOCO_PATH} \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d ${MUJOCO_PATH} \
    && rm mujoco.zip
# Install mujoco-py
RUN git clone https://github.com/openai/mujoco-py.git /mujoco-py \
    && pip install -e /mujoco-py \
    && rm -rf ~/.cache/pip
# Symlink MuJoCo licence
RUN mkdir -p /MJKEY/ \
    && ln -s /MJKEY/mjkey.txt ${MUJOCO_PATH}/mjkey.txt
# Make sure 'glfw.py' can load shared lib
RUN cd ${MUJOCO_PATH}/mjpro150/bin \
    && cp *.so /usr/local/lib/

# Install gym
RUN git clone https://github.com/openai/gym.git /gym \
    && pip install -e '/gym[all]' \
    && rm -rf ~/.cache/pip

ENV CODE_DIR /code
# Create empty directories that will be mounted at runtime
RUN mkdir -p $CODE_DIR/imitation
RUN mkdir -p $CODE_DIR/launchers
RUN mkdir -p $CODE_DIR/data/
RUN mkdir -p $CODE_DIR/DEMOS/
# Define working directory
WORKDIR $CODE_DIR

# Expose tensorboard port
EXPOSE 6006

# Define the command the container executes by default when you launch the built image
CMD ["/usr/bin/pytest"]
