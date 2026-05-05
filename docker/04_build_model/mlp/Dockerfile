FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    tensorflow-cpu==2.16.1

COPY build_mlp.py .

ENV PROCESSED_DIR=/data/processed
ENV MODEL_DIR=/data/model

CMD ["python", "build_mlp.py"]
