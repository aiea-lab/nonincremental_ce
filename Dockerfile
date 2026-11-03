# Image used for the experiments of the paper. It provides torch 1.14.0a0+44dac51
# and torchvision 0.15.0a
FROM nvcr.io/nvidia/pytorch:23.01-py3

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
