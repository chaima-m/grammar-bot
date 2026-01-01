# Use Python 3.9
FROM python:3.9

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the files (including the zip and server.py)
COPY . .

# Create a writable cache directory for the AI models
# (Hugging Face requires this permission fix)
RUN mkdir -p /code/cache && chmod -R 777 /code/cache
ENV SENTENCE_TRANSFORMERS_HOME=/code/cache

# Run the server on Port 7860 (Hugging Face specific port)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]