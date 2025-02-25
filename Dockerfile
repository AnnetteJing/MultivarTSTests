FROM continuumio/miniconda3

# Set working directory inside the container
WORKDIR /scripts

# Create Conda environment
COPY environment.yaml /
RUN conda env create -f /environment.yaml

# Activate Conda environment
SHELL ["conda", "run", "-n", "py3.11-test", "/bin/bash", "-c"]
# RUN echo "source activate py3.11-test" > ~/.bashrc
# ENV PATH /opt/conda/envs/py3.11-test/bin:$PATH

# Copy relevant files & directories
COPY scripts/ /scripts/
COPY src/ /src/
COPY __init__.py /

# Add parent directory of src to PYTHONPATH
ENV PYTHONPATH=/:$PYTHONPATH

# Use Python with the Conda environment
ENTRYPOINT ["conda", "run", "-n", "py3.11-test", "python", "-u"]

# Allow passing additional arguments to docker run
CMD []