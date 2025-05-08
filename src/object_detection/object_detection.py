import torch
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI
from ray import serve
from ray.serve.handle import DeploymentHandle
from RealESRGAN import RealESRGAN
import requests
import os
app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle: DeploymentHandle, image_enhancer_handle: DeploymentHandle):
        self.handle = object_detection_handle
        self.enhancer_handle = image_enhancer_handle

    @app.get(
        "/detect_and_enhance",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect(self):
        image_url = "https://raw.githubusercontent.com/ai-forever/Real-ESRGAN/main/inputs/lr_lion.png"
        enhanced_image = await self.enhancer_handle.enhance.remote(image_url)
        image = await self.handle.detect_after_enhance.remote(enhanced_image)

        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return Response(content=file_stream.getvalue(), media_type="image/jpeg")

    @app.get(
        "/detect",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect(self):

        image_url = "https://raw.githubusercontent.com/ai-forever/Real-ESRGAN/main/inputs/lr_lion.png"
        image = await self.handle.detect.remote(image_url)
        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return Response(content=file_stream.getvalue(), media_type="image/jpeg")



@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    #autoscaling_config={"min_replicas": 1, "max_replicas": 2},
    num_replicas=2,
)
# num of cpu cores used = num_cpus * num_replicas
class PreprocessImage:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = RealESRGAN(device, scale=4)
        self.model.load_weights('weights/RealESRGAN_x4.pth', download=True)

    def enhance(self, image_url: str):
        image = Image.open(requests.get(image_url, stream=True).raw)
        sr_image = self.model.predict(image)
        return sr_image


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    #autoscaling_config={"min_replicas": 1, "max_replicas": 2},
    num_replicas=2,
)
class ObjectDetection:
    def __init__(self):

        # Force YOLOv5 to skip CUDA device detection internally
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""

        # Load YOLOv5 model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

        # Move it to proper device manually
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect_after_enhance(self, image: Image.Image):
        result_im = self.model(image)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))

    def detect(self, image_url: str):
        result_im = self.model(image_url)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))

entrypoint = APIIngress.bind(ObjectDetection.bind(), PreprocessImage.bind())