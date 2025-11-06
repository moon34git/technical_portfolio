# 베이스 이미지: NVIDIA CUDA 11.4 + cuDNN8 + Ubuntu 20.04
FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# 필수 유틸리티 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget vim \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Jupyter 및 PyTorch 환경 (CUDA 11.4 호환 버전)
RUN pip install jupyter torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu114

# 컨테이너 실행 시 진입점
CMD ["bash"]
