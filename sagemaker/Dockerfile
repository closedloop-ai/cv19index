FROM python:3

RUN mkdir -p /opt/cv19index/
COPY sagemaker/docker-requirements.txt /opt/cv19index/docker-requirements.txt
# Provides a cache stage with pandas and xgboost installed
RUN pip install -r /opt/cv19index/docker-requirements.txt

COPY . /opt/cv19index/
# This exposes a serve command for sagemaker
RUN pip install /opt/cv19index/