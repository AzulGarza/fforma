IMAGE := ensambler
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/fforma \
	-w /fforma

init:
	docker build . -t ${IMAGE}

jupyterlab: .require-port
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p $(port):8888 $(IMAGE) \
			bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

.require-port:
ifndef port
	$(error port is required)
endif
