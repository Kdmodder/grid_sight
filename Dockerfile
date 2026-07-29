# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-ui.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements-ui.txt


# =========================
# Stage 2: Runtime
# =========================
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

COPY app.py .
COPY test_data.csv .


RUN useradd --create-home --uid 10001 gridsight \
    && chown -R gridsight:gridsight /app

USER gridsight

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
