# 1. Base Image: Lightweight Python 3.10
FROM python:3.11-slim

# 2. Set Environment Variables
# Prevents Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures console output is not buffered by Docker
ENV PYTHONUNBUFFERED=1
# Configure Streamlit to run headlessly on port 8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install System Dependencies (Required for some math/data libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy Dependencies & Install
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the Application Code
COPY . /app/

# 7. Security: Create a non-root user to run the app
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# 8. Expose the Streamlit Port
EXPOSE 8501

# 9. Healthcheck (Tells Kubernetes if the pod is actually alive)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Execution Command
CMD ["streamlit", "run", "app.py"]