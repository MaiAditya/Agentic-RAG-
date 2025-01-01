# Additional Makefile targets
.PHONY: build run clean deploy

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build --platform linux/amd64 -t pdf-app .
	@echo "Docker image built successfully"

# Run Docker container
run:
	@echo "Starting Docker container..."
	docker run --platform linux/amd64 -p 8000:8000 \
		--env-file .env \
		-v "$(shell pwd)/app:/app/app" \
		-v "$(shell pwd)/chroma_db:/app/chroma_db" \
		-v "$(shell pwd)/logs:/app/logs" \
		pdf-app

# Clean up Docker resources and local directories
clean:
	@echo "Cleaning up Docker images, containers, and Poetry cache..."
	docker rm -f $$(docker ps -a -q) || true
	docker rmi -f $$(docker images -q) || true
	rm -rf .venv
	rm -rf "$(shell pwd)/chroma_db/*"
	rm -rf "$(shell pwd)/logs/*"

# Build and run in sequence
deploy: clean build run
