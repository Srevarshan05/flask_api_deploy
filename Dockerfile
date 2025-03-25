# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for port (Hugging Face default)
ENV PORT=7860

# Run Gunicorn with the Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "0", "app:app"]