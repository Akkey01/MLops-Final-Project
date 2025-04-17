# This file is used to generate the following output for the report.
# 1. Model Size
# 2. Throughout for Batch Inference
# 3. Latency for online (Single sample)
# 4. Concurrency requirements for not on device deployments
# 5. Convert Model to ONNX version
# 6. System Level Optimisation to support required level of optimisation
import torch

# Load the model from .pth file
model_path = ""
device = torch.device("cpu")
# weights_only=False indicates that the model file contains both the model architecture 
# and its weights (not just state_dict). This is important for exporting to ONNX.
model = torch.load(model_path, map_location=device, weights_only=False)
onnx_model_path = "models/IMP.onnx"
