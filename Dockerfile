# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files to avoid caching issues
COPY requirements.txt* /app/

# Install dependencies (supports both Poetry and pip)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set default command (optional)
CMD ["python"]