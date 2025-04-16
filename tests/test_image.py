import requests

# Sample image URL for testing
image_url = "https://www.caloriesecrets.net/wp-content/uploads/2018/05/apples-vs-bananas.jpg"

# Send request to the object detection service
resp = requests.get(f"http://127.0.0.1:8000/detect?image_url={image_url}")

# Save the annotated image with detected objects
with open("output.jpeg", 'wb') as f:
    f.write(resp.content)