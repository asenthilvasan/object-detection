import requests
import time
from pathlib import Path

# Sample image URL for testin
image_url = "https://ultralytics.com/images/zidane.jpg"
local_image_path = Path(__file__).parent / "test_image.jpg"


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
        resp = requests.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")
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
            "http://127.0.0.1:8000/detect-upload",
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
                "http://127.0.0.1:8000/detect-upload",
                files={"file": ("image.jpg", image_bytes, "image/jpeg")}
            )
    else:
        session = requests.Session()
        def do_request():
            return session.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")

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


if __name__ == "__main__":
    run_rps(duration_s=30, rps=5.0, use_upload=True, warmup=5)
