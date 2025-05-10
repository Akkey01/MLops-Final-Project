import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

def benchmark_onnx_text_model(onnx_model_path, tokenizer_path, prompt_text="Once upon a time", batch_size=8):
    # Load ONNX model and tokenizer
    ort_session = ort.InferenceSession(onnx_model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    model_size = os.path.getsize(onnx_model_path)
    print(f"Execution provider: {ort_session.get_providers()}")
    print(f"Model Size: {model_size / 1e6:.2f} MB")

    # Prepare single sample
    inputs = tokenizer(prompt_text, return_tensors="np", padding="longest")
    seq_len = inputs["input_ids"].shape[1]
    input_feed = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "position_ids": np.arange(seq_len)[None, :].astype("int64")  # shape (1, seq_len)
    }

    # Latency benchmark
    latencies = []
    for _ in range(100):
        start = time.time()
        ort_session.run(None, input_feed)
        latencies.append(time.time() - start)

    print(f"Latency (P50): {np.percentile(latencies, 50)*1000:.2f} ms")
    print(f"Latency (P95): {np.percentile(latencies, 95)*1000:.2f} ms")
    print(f"Latency (P99): {np.percentile(latencies, 99)*1000:.2f} ms")
    print(f"Throughput (prompts/sec): {100/np.sum(latencies):.2f}")

    # Prepare batch input
    batch_prompts = [prompt_text] * batch_size
    batch_inputs = tokenizer(batch_prompts, return_tensors="np", padding="longest")
    seq_len = batch_inputs["input_ids"].shape[1]
    batch_input_feed = {
        "input_ids": batch_inputs["input_ids"],
        "attention_mask": batch_inputs["attention_mask"],
        "position_ids": np.tile(np.arange(seq_len), (batch_size, 1)).astype("int64")
    }

    # Batch throughput
    batch_times = []
    for _ in range(50):
        start = time.time()
        ort_session.run(None, batch_input_feed)
        batch_times.append(time.time() - start)

    tokens = batch_inputs["input_ids"].shape[0] * batch_inputs["input_ids"].shape[1] * len(batch_times)
    print(f"Token throughput: {tokens / sum(batch_times):.2f} tokens/sec")

# Run benchmark
benchmark_onnx_text_model(
    "./tinyllama11b_chat_ft1/onnx/model.onnx",
    "./tinyllama11b_chat_ft1",
    prompt_text="What is AI?",
    batch_size=8
)
