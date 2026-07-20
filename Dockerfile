FROM 390402534126.dkr.ecr.us-east-1.amazonaws.com/mlflow-pytorch@sha256:f132b4b813bb24205f9bb5d1e6df7863ba7b852ed42f5886de4e770d0763cb47




# 3) If psycopg2-binary isn’t in requirements.txt, install it here
#    (or add it to requirements.txt and delete this)

# Copy source code
COPY btc_forecast ./btc_forecast
COPY config ./config
COPY logs ./logs
COPY generate_predictions.py .
COPY train_models.py .
COPY model_info.py .
COPY logger.py .
COPY utils ./utils

# Default command
CMD ["python", "generate_predictions.py", "--model_name=LSTMModel"]
