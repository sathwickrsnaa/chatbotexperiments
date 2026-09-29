FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt constraints.txt ./
RUN pip install -c constraints.txt -r requirements.txt && useradd --uid 10001 --create-home appuser \
    && mkdir -p /data && chown appuser:appuser /data
COPY --chown=appuser:appuser transaction_chat ./transaction_chat
COPY --chown=appuser:appuser app.py ./app.py
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
