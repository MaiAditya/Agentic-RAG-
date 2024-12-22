.PHONY: build run stop clean logs restart shell

# Docker image and container configuration
IMAGE_NAME = multimodal-pdf-pipeline
CONTAINER_NAME = multimodal-pdf-app
PORT = 8000

# Build the Docker image
build:
	@echo "Cleaning up old containers and images..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker rmi $(IMAGE_NAME) 2>/dev/null || true
	@echo "Building new image..."
	docker build --no-cache -t $(IMAGE_NAME) .

# Run the container
run:
	@mkdir -p $(PWD)/chroma_db
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):$(PORT) \
		--env-file .env \
		-v $(PWD)/chroma_db:/app/chroma_db \
		$(IMAGE_NAME)

# Stop the container
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Clean everything
clean:
	docker stop $(CONTAINER_NAME) 2>/dev/null || true
	docker rm $(CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(IMAGE_NAME) 2>/dev/null || true
	rm -rf $(PWD)/chroma_db

# Show logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Restart the application
restart: stop run

# Shell into the container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Default target
all: build run logs