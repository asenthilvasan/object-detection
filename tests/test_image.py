import requests
import time

# Sample image URL for testing
image_url = "https://ultralytics.com/images/zidane.jpg"


def write_to_file(image_url: str):
    # Send request to the object detection services
    resp = requests.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")

    # Save the annotated image with detected objects
    with open("output.jpeg", 'wb') as f:
        f.write(resp.content)

def benchmark(n=10):
    times = []
    for _ in range(n):
        start = time.time()
        resp = requests.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Request took {elapsed:.3f}s")
    print(f"\nAvg: {sum(times)/n:.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")

benchmark(50)