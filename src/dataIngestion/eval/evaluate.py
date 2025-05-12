import pandas as pd
from rouge_score import rouge_scorer
import bert_score

def evaluate(df, save_path):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    df["rougeL"] = [
        rouge.score(pred, ref)["rougeL"].fmeasure
        for pred, ref in zip(df["predicted"], df["answer"])
    ]
    _, _, f1 = bert_score.score(df["predicted"].tolist(), df["answer"].tolist(), lang="en")
    df["bertscore_f1"] = f1.tolist()
    df.to_csv(save_path, index=False)
    return df["rougeL"].mean(), df["bertscore_f1"].mean()
