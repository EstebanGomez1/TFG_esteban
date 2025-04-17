FROM nvcr.io/nvidia/pytorch:24.05-py3
ENV FORCE_CUDA="1"

# Timezone. Avoid user interaction.
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add user.
ENV USER=myuser
ENV UID=1000
ENV GID=1000

RUN groupadd -g $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

# Install Python and some utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake

# Copy requirements.txt and install Python packages.
USER $USER
WORKDIR /.cache


USER root
RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 uninstall -y flash-attn || true
RUN pip3 install flash-attn --no-build-isolation
RUN pip3 install torch_scatter
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade pip

USER $USER
RUN pip install 'opencv-python<4.9'
RUN pip install --upgrade timm
RUN pip install "numpy<2"

# Change terminal color
ENV TERM=xterm-256color
RUN echo "PS1='\[\e[1;91m\]\u\[\e[1;37m\]@\[\e[1;93m\]\h\[\e[1;37m\]:\[\e[1;35m\]\w\[\e[1;37m\] → \[\e[0m\]'" >> ~/.bashrc


WORKDIR /workspace

CMD [ "/bin/bash" ]