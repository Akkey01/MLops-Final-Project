def should_use_gpu(data):

    return data["useGpu"] == True

async def route_inference(data):
    if should_use_gpu(data):
        response = await call_gpu_inference_service(data)
    else:
        response = await call_cpu_inference_service(data)
    return response