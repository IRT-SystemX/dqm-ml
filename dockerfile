FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y wget curl # Installe wget et curl

# Set Working dir

WORKDIR /DQM_LIB

# Copy required file for dqm package installation

COPY pyproject.toml .
COPY README.md .
COPY requirements.txt .
COPY dqm ./dqm

# Copy pretrained model in container cache

RUN mkdir -p /root/.cache/torch/hub/checkpoints
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/resnet18-f37072fd.pth
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
RUN wget -P /root/.cache/torch/hub/checkpoints/ https://download.pytorch.org/models/vgg16-397923af.pth


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m pip install .

ADD dqm/main.py .

ENTRYPOINT python main.py \
    --pipeline_config_path "/tmp"/in/"$PIPELINE_CONFIG_PATH" \
    --result_file_path "/tmp"/out/"$RESULT_FILE_PATH" 


