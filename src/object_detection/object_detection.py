import ray
import time
import torch
import asyncio
from PIL import Image
import numpy as np
from io import BytesIO
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File

from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.metrics import Counter, Histogram

# Ray initialization is handled by run_serve.py in Docker
# or automatically by serve run locally

app = FastAPI()


@serve.deployment(num_replicas=1, max_ongoing_requests=100)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle: DeploymentHandle):
        self.handle = object_detection_handle
        self.loop = asyncio.get_running_loop()
        self._handle_roundtrip_hist = Histogram(
            "od_handle_roundtrip_ms",
            description="Full .remote() round-trip: Ray serialization + batch wait + model + return (ms)",
            boundaries=[25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
        )
        self._jpeg_encode_hist = Histogram(
            "od_jpeg_encode_ms",
            description="Response JPEG encoding time in APIIngress (ms)",
            boundaries=[5, 10, 25, 50, 100, 250, 500, 1000],
        )

    def _encode_jpeg(self, image) -> bytes:
        file_stream = BytesIO()
        image.save(file_stream, "jpeg")
        return file_stream.getvalue()

    @app.get(
        "/detect",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect(self, image_url: str):
        t0 = time.perf_counter()
        image = await self.handle.detect.remote(image_url)
        t1 = time.perf_counter()
        content = await self.loop.run_in_executor(None, self._encode_jpeg, image)
        t2 = time.perf_counter()
        self._handle_roundtrip_hist.observe((t1 - t0) * 1000)
        self._jpeg_encode_hist.observe((t2 - t1) * 1000)
        return Response(content=content, media_type="image/jpeg")

    @app.post(
        "/detect-upload",
        responses={200: {"content": {"image/jpeg": {}}}},
        response_class=Response,
    )
    async def detect_upload(self, file: UploadFile = File(...)):
        image_bytes = await file.read()
        t0 = time.perf_counter()
        image = await self.handle.detect_bytes.remote(image_bytes)
        t1 = time.perf_counter()
        content = await self.loop.run_in_executor(None, self._encode_jpeg, image)
        t2 = time.perf_counter()
        roundtrip_ms = (t1 - t0) * 1000
        encode_ms = (t2 - t1) * 1000
        self._handle_roundtrip_hist.observe(roundtrip_ms)
        self._jpeg_encode_hist.observe(encode_ms)
        return Response(content=content, media_type="image/jpeg")


@serve.deployment(
    ray_actor_options={"num_cpus": 1, "num_gpus": 1},
    health_check_period_s=60,
    health_check_timeout_s=30,
    max_ongoing_requests=100,
)
class ObjectDetection:
    def __init__(self):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loop = asyncio.get_running_loop()
        self._preprocess_hist = Histogram(
            "od_preprocess_ms",
            description="CPU time: JPEG decode via PIL per batch (ms)",
            boundaries=[5, 10, 25, 50, 100, 250, 500, 1000],
        )
        self._inference_hist = Histogram(
            "od_inference_ms",
            description="Model forward pass per batch: includes letterbox resize, GPU inference, NMS (ms)",
            boundaries=[25, 50, 100, 250, 500, 1000, 2000, 5000],
        )
        self._postprocess_hist = Histogram(
            "od_postprocess_ms",
            description="CPU time: render bounding boxes + array-to-PIL per batch (ms)",
            boundaries=[5, 10, 25, 50, 100, 250, 500],
        )
        self._batch_size_hist = Histogram(
            "od_batch_size",
            description="Number of images per batch",
            boundaries=[1, 2, 4, 8, 16, 24, 32],
        )
        self._gpu_active_ms = Counter(
            "od_gpu_active_ms_total",
            description="Cumulative GPU inference time (ms). rate()/10 = GPU duty cycle %.",
        )

        print(f"STARTUP: batch_wait=0.01s, max_concurrent_batches=1, max_batch=10, max_ongoing=100, device={self.device}")

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.01)
    async def detect(self, image_urls: list[str]):
        return await self.loop.run_in_executor(None, self._run_detect, image_urls)

    def _run_detect(self, image_urls: list[str]):
        batch_size = len(image_urls)
        self._batch_size_hist.observe(batch_size)

        # Stage 1: no separate preprocess — YOLOv5 downloads/decodes URLs internally

        # Stage 2: model forward (includes URL fetch + resize + GPU + NMS)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        results = self.model(image_urls)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Stage 3: CPU postprocess (render bounding boxes + convert to PIL)
        rendered = results.render()
        output = [Image.fromarray(im.astype(np.uint8)) for im in rendered]
        t2 = time.perf_counter()

        inf_ms = (t1 - t0) * 1000
        post_ms = (t2 - t1) * 1000
        self._inference_hist.observe(inf_ms)
        self._postprocess_hist.observe(post_ms)
        self._gpu_active_ms.inc(inf_ms)

        print(f"STAGES batch={batch_size} | inference={inf_ms:.1f}ms | postprocess={post_ms:.1f}ms")
        return output

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.01)
    async def detect_bytes(self, image_bytes_list: list[bytes]):
        return await self.loop.run_in_executor(None, self._run_detect_bytes, image_bytes_list)

    def _run_detect_bytes(self, image_bytes_list: list[bytes]):
        batch_size = len(image_bytes_list)
        self._batch_size_hist.observe(batch_size)

        # Stage 1: CPU JPEG decode — PIL opens each image from raw bytes
        t0 = time.perf_counter()
        images = [Image.open(BytesIO(b)) for b in image_bytes_list]
        t1 = time.perf_counter()

        # Stage 2: model forward — YOLOv5 letterbox resize + GPU inference + NMS
        # torch.cuda.synchronize() ensures GPU work is complete before we stop the timer.
        # Without it, CUDA ops are async and the timer would only measure kernel launch time.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        results = self.model(images)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Stage 3: CPU postprocess — draw bounding boxes on images + convert to PIL
        rendered = results.render()
        output = [Image.fromarray(im.astype(np.uint8)) for im in rendered]
        t3 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000
        inf_ms = (t2 - t1) * 1000
        post_ms = (t3 - t2) * 1000
        self._preprocess_hist.observe(pre_ms)
        self._inference_hist.observe(inf_ms)
        self._postprocess_hist.observe(post_ms)
        self._gpu_active_ms.inc(inf_ms)

        print(
            f"STAGES batch={batch_size}"
            f" | preprocess={pre_ms:.1f}ms"
            f" | inference={inf_ms:.1f}ms"
            f" | postprocess={post_ms:.1f}ms"
            f" | total={(t3 - t0) * 1000:.1f}ms"
        )
        return output


entrypoint = APIIngress.bind(ObjectDetection.bind())