# syntax=docker/dockerfile:1.2
FROM python:3.12.10

# Set the working directory to /app
WORKDIR /app

# Copy all files
COPY . .

# Instalar uv
RUN pip install uv

# Instalar dependencias usando uv
RUN uv sync

# Set the working directory to /app
WORKDIR /app/src

# Expose port 8080, which is the port by default
#EXPOSE 8080

# Command to run the application when the container starts
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]