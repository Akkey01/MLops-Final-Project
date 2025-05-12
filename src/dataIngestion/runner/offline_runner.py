import pandas as pd
from eval.evaluate import evaluate

df = pd.read_csv("test_data/offline_test.csv")
avg_rouge, avg_bertscore = evaluate(df, "test_data/results_offline.csv")
print(f"Offline Evaluation: ROUGE-L={avg_rouge:.2f}, BERTScore={avg_bertscore:.2f}")
