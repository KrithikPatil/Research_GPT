# Use an official Python runtime as a parent image
FROM python:3.11.3-slim

# Set the working directory in the container
WORKDIR /Research_GPT

# Copy the requirements file into the container
COPY requirements.txt .

RUN pip install --no-cache-dir virtualenv && python -m venv research_gpt

# Activate the virtual environment and install dependencies
RUN /bin/bash -c "source /Research_GPT/research_gpt/bin/activate && pip install --no-cache-dir -r requirements.txt"

# Step 7: Set environment variable to use the virtual environment
ENV PATH="/Research_GPT/research_gpt/bin:$PATH"

COPY distilbert_model faiss_index.index processed_springer_papers_DL.json /Research_GPT/

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV KMP_DUPLICATE_LIB_OK='True'

# Run the application using the virtual environment (activated environment)
CMD ["python", "app.py"]