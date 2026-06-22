FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN pip install --no-cache-dir \
    dvc[s3]==3.40.0 \
    pathspec==0.11.2 \
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0 \
    evidently==0.4.30
COPY Multimodal/drift_monitor/drift_monitor.py .
# No s3_utils

ENV EVAL_DIR=/app/data/eval
ENV PRODUCTION_LOG_DIR=/app/data/production_logs
ENV DRIFT_REPORT_DIR=/app/data/xai/drift_reports
ENV CHECK_INTERVAL_HOURS=24
ENV PSI_THRESHOLD=0.25
ENV KS_P_THRESHOLD=0.05

COPY .dvc/ .dvc/
CMD ["python", "drift_monitor.py"]
