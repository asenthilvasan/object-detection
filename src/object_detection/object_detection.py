import ray
import torch
import asyncio
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File

from ray import serve
from ray.serve.handle import DeploymentHandle

# Ray initialization is handled by run_serve.py in Docker
# or automatically by serve run locally

app = FastAPI()


@serve.deployment(num_replicas=1, max_ongoing_requests=100)
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

    @app.post(
        "/detect-upload",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect_upload(self, file: UploadFile = File(...)):
        image_bytes = await file.read()
        image = await self.handle.detect_bytes.remote(image_bytes)
        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return Response(content=file_stream.getvalue(), media_type="image/jpeg")


@serve.deployment(
    ray_actor_options={"num_cpus": 2, "num_gpus": 1},
    health_check_period_s=60,
    health_check_timeout_s=30,
    max_ongoing_requests=100,
    #autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    # num of cpu cores used = num_cpus * num_replicas
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loop = asyncio.get_running_loop()

    
    
    # max_concurrent_batches=2: while one batch runs in the thread pool, the event loop
    # can accumulate a second batch simultaneously — pipelines GPU work and reduces idle time.
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.5, max_concurrent_batches=2)
    async def detect(self, image_urls: list[str]):
        # run_in_executor keeps the async event loop free while inference runs in a thread.
        # This is required for max_concurrent_batches > 1 to work — without it, the event
        # loop blocks and the second batch cannot accumulate while the first one executes.
        return await self.loop.run_in_executor(None, self._run_detect, image_urls)

    def _run_detect(self, image_urls: list[str]):
        print(f"DEBUG: Processing batch of size {len(image_urls)}")
        results = self.model(image_urls)
        return [Image.fromarray(im.astype(np.uint8)) for im in results.render()]

    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.5, max_concurrent_batches=2)
    async def detect_bytes(self, image_bytes_list: list[bytes]):
        return await self.loop.run_in_executor(None, self._run_detect_bytes, image_bytes_list)

    def _run_detect_bytes(self, image_bytes_list: list[bytes]):
        print(f"DEBUG: Processing batch of size {len(image_bytes_list)}")
        images = [Image.open(BytesIO(b)) for b in image_bytes_list]
        results = self.model(images)
        return [Image.fromarray(im.astype(np.uint8)) for im in results.render()]


entrypoint = APIIngress.bind(ObjectDetection.bind())