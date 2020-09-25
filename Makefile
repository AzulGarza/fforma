IMAGE := ensambler
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/fforma \
	-w /fforma

init:
	docker build . -t ${IMAGE}

jupyterlab:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
			bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

jlserver: .require-instance .require-dir
	ssh ${instance} "cd ${dir} && make jupyterlab -e PORT=${PORT}"

.require-instance:
ifndef instance
	$(error instance is required)
endif

.require-dir:
ifndef dir
	$(error dir is required)
endif
