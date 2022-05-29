FROM python:3.8-slim-bullseye
WORKDIR /usr/src/kmeans_homemade
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN pip install -e .
