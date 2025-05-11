import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Model path
MODEL_PATH = Path("./backend/tinyllama11b_chat_ft1")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH.as_posix(), local_files_only=True)
model.eval()

# Sample input
sample_text = "Once upon a time"
inputs = tokenizer(sample_text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Export to ONNX
torch.onnx.export(
    model,
    args=(input_ids,),
    f=MODEL_PATH / "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
    opset_version=13,
    do_constant_folding=True
)

print(f"Model exported to {MODEL_PATH / 'model.onnx'}")
