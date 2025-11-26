FROM python:3.10-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY src/requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY src /app/src
COPY models /app/models
COPY .env /app/.env

# Expose the port Hugging Face uses
EXPOSE 7860

# Start FastAPI using uvicorn
# main.py contains `app = FastAPI()`
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]
