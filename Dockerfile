FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.4

COPY pyproject.toml README.md ./
COPY src ./src
COPY model_impl.py app.py train.py evaluate.py ./
COPY My First Project.v1i.yolov8 ./My First Project.v1i.yolov8

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN mkdir -p /app/data /app/models

CMD ["python", "app.py", "--mode", "video", "--source", "data/input.mp4", "--output", "data/output.mp4"]
