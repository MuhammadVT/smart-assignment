# The hub is a single static page served by a stdlib Python script, so the image
# needs nothing but a Python base and the two files.
#
# A Dockerfile (rather than leaving it to Nixpacks auto-detection) is deliberate:
# this directory lives inside a repo whose root is a different Python project, and
# an explicit build removes any doubt about what Railway is building.
FROM python:3.12-slim

WORKDIR /app
COPY index.html serve.py ./

# Railway injects $PORT at runtime; serve.py falls back to 3000 for local runs.
EXPOSE 3000
CMD ["python3", "serve.py"]
