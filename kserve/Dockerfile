FROM python:3.11-slim

WORKDIR /app

COPY kserve/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["python", "kserve/kserve_server.py"]
