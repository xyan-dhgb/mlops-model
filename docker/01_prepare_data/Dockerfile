FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    gdown==5.2.0 \
    h5py==3.11.0 \
    pandas==2.2.2

COPY download_dataset.py .

# DATA_DIR và GDRIVE_FILE_ID được truyền qua biến môi trường khi chạy
ENV DATA_DIR=/data/raw
ENV GDRIVE_FILE_ID=""

CMD ["python", "download_dataset.py"]
