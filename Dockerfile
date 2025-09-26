FROM registry-docker-registry.registry.svc.cluster.local:5000/mlflow-pytorch:latest


# 2) App requirements
#    (copy requirements first to leverage cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) If psycopg2-binary isn’t in requirements.txt, install it here
#    (or add it to requirements.txt and delete this)
RUN pip install psycopg2-binary

# Copy source code
COPY btc_forecast ./btc_forecast
COPY config ./config
COPY logs ./logs
COPY generate_predictions.py .
COPY train_models.py .
COPY metadata.py .
COPY utils.py .
COPY download_artifacts.py .

# Default command
CMD ["python", "generate_predictions.py", "--model_name=LSTMModel"]
