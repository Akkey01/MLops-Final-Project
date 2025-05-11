from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'mlops_user',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Define the DAG
with DAG(
    dag_id='export_and_benchmark_onnx_model',
    default_args=default_args,
    description='Export ONNX model and benchmark it using Airflow',
    start_date=datetime(2024, 5, 1),
    schedule_interval='@daily',  # can be changed to manual or cron
    catchup=False
) as dag:

    # Task 1: Export model to ONNX
    export_model = BashOperator(
        task_id='export_model_to_onnx',
        bash_command='python ./backend/export_to_onnx.py',
    )

    # Task 2: Benchmark the exported ONNX model
    benchmark_model = BashOperator(
        task_id='benchmark_onnx_model',
        bash_command='python ./backend/benchmark_onnx.py',
    )

    # Define task dependencies
    export_model >> benchmark_model
