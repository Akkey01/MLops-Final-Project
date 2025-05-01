from hardware_optimized import getInferenceOnGPU
from dataIngestion.multimediaHandler import MultiMediaHandler

def should_use_gpu(data):

    return data["useGpu"] == True

async def route_inference(data, file):
    if should_use_gpu(data):
        response = await getInferenceOnGPU(file)
    else:
        handler = MultiMediaHandler(file)
        respone = handler.process()
        
    return response