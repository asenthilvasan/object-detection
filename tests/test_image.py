import requests
import asyncio
import aiohttp
import time
import os
from pathlib import Path

# Sample image URL for testin
image_url = "https://ultralytics.com/images/zidane.jpg"
base_url = os.environ.get("OD_BASE_URL", "http://127.0.0.1:8000")
_default_dir = Path(__file__).parent
_writable_dir = Path(os.environ.get("TEST_IMAGE_DIR", "/tmp"))
local_image_path = (_writable_dir if _writable_dir.exists() and os.access(_writable_dir, os.W_OK) else _default_dir) / "test_image.jpg"


def download_test_image():
    #download image once
    if not local_image_path.exists():
        print("Downloading test image...")
        resp = requests.get(image_url)
        local_image_path.write_bytes(resp.content)
        print(f"Saved to {local_image_path}")
    return local_image_path


def benchmark_url(n=10):
    #Benchmark for fetching image every time
    print(f"\nURL Endpoint Benchmark (n={n})")
    times = []
    for i in range(n):
        start = time.time()
        resp = requests.get(f"{base_url}/detect?image_url={image_url}")
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"[{i+1}/{n}] {elapsed:.3f}s")
    print(f"Avg: {sum(times)/n:.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    return times


def benchmark_upload(n=10):
    #Benchmark for just uploading the imaga data and fetching once
    print(f"\nUpload Endpoint Benchmark (n={n})")
    image_path = download_test_image()
    image_bytes = image_path.read_bytes()
    
    # Use session to reuse TCP connection 
    session = requests.Session()
    
    times = []
    for i in range(n):
        start = time.time()
        resp = session.post(
            f"{base_url}/detect-upload",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")}
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"[{i+1}/{n}] {elapsed:.3f}s")
    print(f"Avg: {sum(times)/n:.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    
    session.close()
    return times


def run_rps(duration_s=30, rps=5.0, use_upload=True, warmup=5):
    # Simple fixed-rate loop to target consistent request rate.
    interval = 1.0 / rps
    times = []

    if use_upload:
        image_path = download_test_image()
        image_bytes = image_path.read_bytes()
        session = requests.Session()
        def do_request():
            return session.post(
                f"{base_url}/detect-upload",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")}
            )
    else:
        session = requests.Session()
        def do_request():
            return session.get(f"{base_url}/detect?image_url={image_url}")

    # Warmup
    for _ in range(warmup):
        do_request()

    start_wall = time.perf_counter()
    next_tick = start_wall
    count = 0
    while True:
        now = time.perf_counter()
        if now >= start_wall + duration_s:
            break
        if now < next_tick:
            time.sleep(next_tick - now)
        req_start = time.perf_counter()
        do_request() 
        elapsed = time.perf_counter() - req_start
        times.append(elapsed)
        count += 1
        next_tick += interval

    session.close()

    if times:
        achieved_rps = count / duration_s
        print(f"\nRPS run: target={rps:.2f}, achieved={achieved_rps:.2f}, requests={count}")
        print(f"Avg: {sum(times)/len(times):.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")
    return times


async def run_rps_concurrent(duration_s=30, rps=5.0, concurrency=64, warmup=5):
    """Same as run_rps but fires requests concurrently so we can actually
    saturate the server instead of being limited by serial round-trip time."""
    interval = 1.0 / rps
    times = []
    sem = asyncio.Semaphore(concurrency)

    image_path = download_test_image()
    image_bytes = image_path.read_bytes()

    async def do_request(session):
        async with sem:
            data = aiohttp.FormData()
            data.add_field("file", image_bytes,
                           filename="image.jpg", content_type="image/jpeg")
            start = time.perf_counter()
            async with session.post(f"{base_url}/detect-upload", data=data) as resp:
                await resp.read()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    async with aiohttp.ClientSession() as session:
        # Warmup
        for _ in range(warmup):
            await do_request(session)
        times.clear()

        # Fire requests at target RPS for duration_s
        tasks = []
        start_wall = time.perf_counter()
        next_tick = start_wall

        while True:
            now = time.perf_counter()
            if now >= start_wall + duration_s:
                break
            if now < next_tick:
                await asyncio.sleep(next_tick - now)
            tasks.append(asyncio.create_task(do_request(session)))
            next_tick += interval

    
        # Wait for all in-flight requests to finish
        await asyncio.gather(*tasks)
        
    total_duration = time.perf_counter() - start_wall

    if times:
        achieved_rps = len(times) / total_duration
        sorted_times = sorted(times)
        p50 = sorted_times[len(times) // 2]
        p95 = sorted_times[int(len(times) * 0.95)]
        p99 = sorted_times[int(len(times) * 0.99)]
        print(f"\nRPS run: target={rps:.2f}, achieved={achieved_rps:.2f}, requests={len(times)}, duration={total_duration:.2f}s")
        print(f"Avg: {sum(times)/len(times):.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")
        print(f"p50: {p50:.3f}s | p95: {p95:.3f}s | p99: {p99:.3f}s")
    return times


if __name__ == "__main__":
    # Lower defaults to debug the 5 RPS latency issue
    rps_levels = [5, 10, 20, 30, 40, 50, 60]

    async def main():
        print(f"Benchmarking against: {base_url}")
        # Single warmup request to wake up connection pool/DNS
        try:
           requests.get(f"{base_url}/detect?image_url={image_url}", timeout=5)
        except:
           pass

        for rps in rps_levels:
            await run_rps_concurrent(duration_s=30, rps=rps, concurrency=64, warmup=2)

    asyncio.run(main())
