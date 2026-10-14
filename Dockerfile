FROM python:3.12-slim
# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /chunking

COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY main_chunking.py utils.py main_API.py ./

# Set environment variables
ENV MODEL_NAME="intfloat/e5-large-v2"


EXPOSE 8084
CMD ["uvicorn", "main_API:app", "--host", "0.0.0.0", "--port", "8084"]
