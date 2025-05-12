from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests

# Simulated Prometheus check
PROMETHEUS_URL = "http://prometheus:9090/api/v1/query"
CANARY_METRIC = 'model_latency_p95_seconds{version="canary"}'
LATENCY_THRESHOLD = 0.3  # seconds

def check_prometheus_metrics():
    response = requests.get(PROMETHEUS_URL, params={"query": CANARY_METRIC})
    result = response.json()

    if not result['data']['result']:
        raise Exception("No metrics returned for canary deployment")

    latency = float(result['data']['result'][0]['value'][1])
    print(f"Canary model P95 latency: {latency:.3f}s")

    if latency > LATENCY_THRESHOLD:
        raise Exception("Canary latency exceeds threshold — trigger rollback")

def rollback_model():
    print("Rolling back to previous stable model...")
    # Add real rollback logic here (e.g., API call, helm rollback, etc.)

default_args = {
    'owner': 'mlops',
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

with DAG(
    dag_id='canary_deployment_pipeline',
    default_args=default_args,
    description='Canary rollout for ONNX model version with Prometheus validation',
    start_date=datetime(2024, 5, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    deploy_canary = BashOperator(
        task_id='deploy_canary_model',
        bash_command='echo "Canary model deployed to 10% of traffic."'
    )

    send_test_requests = BashOperator(
        task_id='send_test_requests',
        bash_command='python ../backend/benchmark_onnx.py'
    )

    validate_canary = PythonOperator(
        task_id='validate_canary_health',
        python_callable=check_prometheus_metrics
    )

    promote_model = BashOperator(
        task_id='promote_canary_model',
        bash_command='echo "Canary model promoted to 100% of traffic."'
    )

    rollback_canary = PythonOperator(
        task_id='rollback_model',
        python_callable=rollback_model,
        trigger_rule='one_failed'  # only run on failure of validate_canary
    )

    deploy_canary >> send_test_requests >> validate_canary >> promote_model
    validate_canary >> rollback_canary
