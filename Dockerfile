# Use an official Python runtime as a parent image
FROM python:3.11.3-slim

# Set the working directory in the container
WORKDIR /Research_GPT

# Copy the requirements file into the container
COPY requirements.txt .

# Install virtualenv (to create a virtual environment)
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment named research_gpt
RUN python -m venv research_gpt

# Activate the virtual environment and install the dependencies into it
RUN . /research_gpt/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV KMP_DUPLICATE_LIB_OK='True'

# Run the application using the virtual environment (activated environment)
CMD ["/research_gpt/bin/python", "app.py"]