from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG("llm_test_pipeline", start_date=datetime(2024, 1, 1), schedule_interval=None, catchup=False) as dag:
    
    invoke = BashOperator(
        task_id="invocation",
        bash_command='echo "invoked"'
    )

    online = BashOperator(
        task_id="online_test",
        bash_command="python3 /opt/airflow/runner/online_runner.py"
    )

    offline = BashOperator(
        task_id="offline_test",
        bash_command="python3 /opt/airflow/runner/offline_runner.py"
    )

    invoke >> online >> offline
