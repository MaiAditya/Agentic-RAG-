FROM python:3.9-bullseye

WORKDIR /app

# Install system dependencies including OpenCV requirements and SQLite3
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install newer version of SQLite3
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3450000.tar.gz && \
    tar xvfz sqlite-autoconf-3450000.tar.gz && \
    cd sqlite-autoconf-3450000 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3450000* && \
    ldconfig

# Install Poetry and add to PATH
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY poetry.lock pyproject.toml ./

# Configure Poetry and pip
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0 \
    LD_LIBRARY_PATH=/usr/local/lib

# Install dependencies with platform specific settings
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    protobuf==3.20.3 && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Download NLTK and spaCy data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('universal_tagset'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')" \
    && python -m spacy download en_core_web_sm

# Create necessary directories
RUN mkdir -p /app/chroma_db /app/logs

# Create and set permissions for ChromaDB directory
RUN mkdir -p /app/chroma_db && chmod 777 /app/chroma_db

# Copy application code and fix device string
COPY app ./app
RUN sed -i 's/device="cpu "/device="cpu"/g' /app/app/database/chroma_client.py

EXPOSE 8000

CMD ["poetry", "run", "python", "-m", "app.main"]


