DOCKER_IMAGE_NAME := aicom
DOCKER_IMAGE_VERSION := 1.0

run:
	@echo "Running a Docker Container"
	docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	-it --rm -v ./:/projects/ai_compiler_study $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION) \
	bash scripts/benchmark.sh

docker_image:
	@echo "Building a Docker Image"
	docker pull nvcr.io/nvidia/pytorch:23.10-py3
	docker build --no-cache --tag $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION) .

clean:
	docker rmi $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_VERSION)
