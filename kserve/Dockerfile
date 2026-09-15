FROM python:3.11-slim

WORKDIR /app

COPY kserve/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY kserve/kserve_server.py /app/kserve/kserve_server.py
COPY predictor /app/predictor
COPY models/champion_model.pkl /app/models/champion_model.pkl

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "/app/kserve/kserve_server.py"]
