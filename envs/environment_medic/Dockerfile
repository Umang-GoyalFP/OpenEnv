ARG BASE_IMAGE=openenv-base:latest
FROM ${BASE_IMAGE}

# Install dependencies
COPY envs/environment_medic/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy core and environment code
COPY src/openenv/core/ /app/src/openenv/core/
COPY envs/environment_medic/ /app/envs/environment_medic/

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "envs.environment_medic.app:app", "--host", "0.0.0.0", "--port", "8000"]
