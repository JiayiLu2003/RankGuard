FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RANKGUARD_ARTIFACT_DIR=/app/artifacts/latest
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .
COPY --from=frontend /app/frontend/dist ./frontend/dist
COPY artifacts/latest ./artifacts/latest
EXPOSE 8000
CMD ["uvicorn", "rankguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
