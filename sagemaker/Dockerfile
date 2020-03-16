FROM ubuntu:latest

RUN apt-get update
RUN apt-get --yes upgrade
RUN apt-get --yes install python3 python3-pip

RUN mkdir -p /opt/cv19index/
COPY sagemaker/docker-requirements.txt /opt/cv19index/docker-requirements.txt
# Provides a cache stage with pandas and xgboost installed
RUN pip3 install -r /opt/cv19index/docker-requirements.txt

COPY . /opt/cv19index/
# This exposes a serve command for sagemaker
RUN pip3 install /opt/cv19index/
