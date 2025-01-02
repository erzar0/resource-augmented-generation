FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code
COPY requirements.txt /code/

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /code/

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8000", "--server.address=0.0.0.0"]
