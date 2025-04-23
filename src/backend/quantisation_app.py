import os
import neural_compressor
from neural_compressor import quantization
from torchvision import datasets, transforms

# Quantization
# Create Test set to test the accuracy for quantized output
fp32_model = neural_compressor.model.onnx_model.ONNXModel("./models/IMP.onnx")

# Configure the quantizer
config_ptq = neural_compressor.PostTrainingQuantConfig(
    accuracy_criterion = neural_compressor.config.AccuracyCriterion(
        criterion="absolute",  
        tolerable_loss=0.05  # We will tolerate up to 0.05 less accuracy in the quantized model
    ),
    approach="static", 
    device='cpu', 
    quant_level=1,
    quant_format="QOperator", 
    recipes={"graph_optimization_level": "ENABLE_EXTENDED"}, 
    calibration_sampling_size=128
)

# Find the best quantized model meeting the accuracy criterion
q_model = quantization.fit(
    model=fp32_model, 
    conf=config_ptq, 
    calib_dataloader=eval_dataloader,
    eval_dataloader=eval_dataloader, 
    eval_metric=neural_compressor.metric.Metric(name='topk')
)

# Save quantized model
q_model.save_model_to_file("./models/IMP_quantized_aggressive.onnx")

#Testing the quantised model version
onnx_model_path = "./models/IMP_quantized_aggressive.onnx"
ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
benchmark_session(ort_session)

def benchmark_session(ort_session):

    print(f"Execution provider: {ort_session.get_providers()}")
    model_size = os.path.getsize(onnx_model_path) 
    print(f"Model Size on Disk: {model_size/ (1e6) :.2f} MB")

    ## Benchmark accuracy
    # Need to write custom logic for testing we aint dealing with images
    correct = 0
    total = 0
    for images, labels in test_loader:
        images_np = images.numpy()
        outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: images_np})[0]
        predicted = np.argmax(outputs, axis=1)
        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum()
    accuracy = (correct / total) * 100

    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    ## Benchmark inference latency for single sample

    num_trials = 100  # Number of trials

    # Get a single sample from the test data

    single_sample, _ = next(iter(test_loader))  
    single_sample = single_sample[:1].numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})

    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_sample})
        latencies.append(time.time() - start_time)

    print(f"Inference Latency (single sample, median): {np.percentile(latencies, 50) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 95th percentile): {np.percentile(latencies, 95) * 1000:.2f} ms")
    print(f"Inference Latency (single sample, 99th percentile): {np.percentile(latencies, 99) * 1000:.2f} ms")
    print(f"Inference Throughput (single sample): {num_trials/np.sum(latencies):.2f} FPS")

    ## Benchmark batch throughput

    num_batches = 50  # Number of trials

    # Get a batch from the test data
    batch_input, _ = next(iter(test_loader))  
    batch_input = batch_input.numpy()

    # Warm-up run
    ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})

    batch_times = []
    for _ in range(num_batches):
        start_time = time.time()
        ort_session.run(None, {ort_session.get_inputs()[0].name: batch_input})
        batch_times.append(time.time() - start_time)

    batch_fps = (batch_input.shape[0] * num_batches) / np.sum(batch_times) 
    print(f"Batch Throughput: {batch_fps:.2f} FPS")