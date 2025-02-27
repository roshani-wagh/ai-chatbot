# Use official Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all files, including .pkl files but excluding files in .dockerignore
COPY . .


# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
