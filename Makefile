IMAGE := ensambler
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PORT := 8888
INSTANCE := mega
JUPYTER_KIND := lab

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/fforma \
	-w /fforma

init:
	docker build . -t ${IMAGE}

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
			bash -c "jupyter ${JUPYTER_KIND} --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

jupyter_server: .require-dir tunnel
	ssh ${INSTANCE} "cd ${dir} && make jupyter -e PORT=${PORT}"

tunnel:
	ssh -NfL localhost:${PORT}:localhost:${PORT} ${INSTANCE}

.require-dir:
ifndef dir
	$(error dir is required)
endif
