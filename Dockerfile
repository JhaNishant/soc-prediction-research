# Use the official Miniconda3 base image
FROM continuumio/miniconda3

# Set environment variables to reduce Python stdout buffering (good for logging)
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Install system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y time && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements files to the working directory
COPY conda-requirements.txt .
COPY requirements.txt .

# Install the conda-required modules
RUN conda install --yes --file conda-requirements.txt && \
    conda clean --all --yes

# Install the pip-required modules using pip
# Running upgrade pip setuptools first to ensure they are up to date before installing the requirements
RUN pip install --upgrade pip setuptools && \
    pip install -r requirements.txt --extra-index-url=https://pypi.nvidia.com

# Copy your project files into the container
COPY . .
