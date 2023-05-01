# Use the official Miniconda3 base image
FROM continuumio/miniconda3

# Copy the conda-requirements.txt file (with your conda-required modules) to the working directory
COPY conda-requirements.txt .

# Install the conda-required modules using conda
RUN conda install --yes --file conda-requirements.txt

# Copy the requirements.txt file (with your pip-required modules) to the working directory
COPY requirements.txt .

# Install the pip-required modules using pip
RUN pip install -r requirements.txt

# Copy your project files into the container
COPY . .
