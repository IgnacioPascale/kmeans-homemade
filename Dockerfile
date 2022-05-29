FROM jupyter/datascience-notebook:latest

WORKDIR /home/jovyan/work
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
