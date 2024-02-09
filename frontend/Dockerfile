FROM python:3.9-slim

WORKDIR /app

COPY requirements/ requirements/

RUN pip install --no-cache-dir -r requirements/base-requirements.txt -r requirements/fe-requirements.txt

COPY .streamlit/ .streamlit/
COPY frontend/ frontend/
COPY lib/ lib/

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

