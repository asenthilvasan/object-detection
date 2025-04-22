import requests
import time

# Sample image URL for testing
image_url = "https://people.com/thmb/oni3ZYC5MJBRdc_dI65sgjnex0s=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(999x0:1001x2)/dog-hug-2f174202e9cf4b36bc6c81b196a6d7bd.jpg"



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