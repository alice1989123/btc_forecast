FROM registry-docker-registry.registry.svc.cluster.local:5000/mlflow-pytorch:latest




# 3) If psycopg2-binary isn’t in requirements.txt, install it here
#    (or add it to requirements.txt and delete this)

# Copy source code
COPY btc_forecast ./btc_forecast
COPY config ./config
COPY logs ./logs
COPY generate_predictions.py .
COPY train_models.py .
COPY metadata.py .
COPY utils .
COPY download_artifacts.py .

# Default command
CMD ["python", "generate_predictions.py", "--model_name=LSTMModel"]
