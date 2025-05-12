import pandas as pd
import requests
import sys
sys.path.append("/opt/airflow")

from eval.evaluate import evaluate

API_URL = "http://backend:8000/generate"

df = pd.read_csv("/opt/airflow/test_data/online_test.csv")

print(df.info())

def query_llm(q):  
    try:
        r = requests.post(API_URL, json={"text": q})
        return r.json().get("output", "") if r.status_code == 200 else ""
    except:
        return "API error"

df["predicted"] = df["question"].apply(query_llm)

avg_rouge, avg_bertscore = evaluate(df, "test_data/results_online.csv")
print(f"Online Evaluation: ROUGE-L={avg_rouge:.2f}, BERTScore={avg_bertscore:.2f}")
