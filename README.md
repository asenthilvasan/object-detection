# Object Detection

A scalable object detection service built with Ray Serve and YOLOv5, developed for ML pipeline research at UCSC.

## What It Does

This project provides a REST API for real-time object detection on images. It uses:

- **YOLOv5** for object detection
- **Ray Serve** for scalable model serving with autoscaling support
- **Prometheus + Grafana** for metrics and monitoring

The service accepts an image URL, runs object detection, and returns the annotated image with bounding boxes.

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation)
- [Grafana](https://grafana.com/) (optional, for dashboards): `brew install grafana`

## Setup

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Run the service:**
   ```bash
   ./start.sh
   ```

   Or run just the Ray Serve application:
   ```bash
   poetry run serve run src.object_detection.object_detection:entrypoint
   ```

## Usage

Send a GET request with an image URL:

```bash
curl "http://localhost:8000/detect?image_url=https://example.com/image.jpg" --output result.jpg
```

## Services

| Service | URL |
|---------|-----|
| Ray Serve API | http://localhost:8000 |
| Ray Dashboard | http://localhost:8265 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
