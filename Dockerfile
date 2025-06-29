# Use official Python image
FROM python:3.10-slim-bullseye

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy files
COPY . /app/

# Install dependencies and update system packages to reduce vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
