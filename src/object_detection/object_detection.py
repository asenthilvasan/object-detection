import ray
import torch
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI

from ray import serve
from ray.serve.handle import DeploymentHandle

# Initialize Ray with dashboard enabled
# Dashboard available at http://localhost:8265
ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle: DeploymentHandle):
        self.handle = object_detection_handle

    @app.get(
        "/detect",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect(self, image_url: str):
        image = await self.handle.detect.remote(image_url)
        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return Response(content=file_stream.getvalue(), media_type="image/jpeg")


@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    #autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    # num of cpu cores used = num_cpus * num_replicas
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect(self, image_url: str):
        result_im = self.model(image_url)
        return Image.fromarray(result_im.render()[0].astype(np.uint8))


entrypoint = APIIngress.bind(ObjectDetection.bind())