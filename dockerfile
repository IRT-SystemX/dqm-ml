FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Copier le fichier local dans le conteneur
COPY * .
COPY dqm ./dqm
RUN python -m pip install .

ADD dqm/main.py .

ENTRYPOINT python main.py \
    --pipeline_config_path "/tmp"/"$PIPELINE_CONFIG_PATH" \
    --result_file_path "/tmp"/"$RESULT_FILE_PATH" 


