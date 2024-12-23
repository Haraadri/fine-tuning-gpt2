# Use the official Python image as the base image
FROM python:3.10.0-slim

# Set the working directory in the container
WORKDIR .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt and the script into the container
COPY requirements.txt /requirements.txt
COPY fine_tuned_gpt_model /fine_tuned_gpt_model
COPY script.py /script.py

# Install the required Python packages
RUN pip install -r requirements.txt

# Expose the port (if serving a model)
EXPOSE 8000

# Set the command to run the script
CMD ["python3", "script.py"]
