docker run --rm \
  -p 8080:8080 \
  -e ENVIRONMENT=prod \
  -e LOG_LEVEL=INFO \
  -e PORT=8080 \
  <image>:<tag>
