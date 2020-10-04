IMAGE := ensambler
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PORT := 8888
INSTANCE := mega
JUPYTER_KIND := lab
EXPERIMENTS_DIR := experiments

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/fforma \
	-w /fforma

init:
	docker build . -t ${IMAGE} && mkdir ${EXPERIMENTS_DIR}

datasets:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python -m fforma.experiments.datasets.main \
							--directory ${EXPERIMENTS_DIR}

base: base_cv base_training

base_cv: .require-dataset
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python -m fforma.experiments.base.main \
							--directory ${EXPERIMENTS_DIR} \
							--dataset ${dataset}

base_training: .require-dataset
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python -m fforma.experiments.base.main \
							--directory ${EXPERIMENTS_DIR} \
							--dataset ${dataset} \
							--training

benchmarks: .require-dataset
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python -m fforma.experiments.benchmarks.main \
							--directory ${EXPERIMENTS_DIR} \
							--dataset ${dataset}

run: .require-dataset .require-model .require-splits .require-trials
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		python -m fforma.experiments.cross_validation.main \
							--directory ${EXPERIMENTS_DIR} \
							--dataset ${dataset} \
							--model ${model} \
							--n_splits ${splits} \
							--n_trials ${trials} \

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

.require-dataset:
ifndef dataset
	$(error dataset is required)
endif

.require-model:
ifndef model
	$(error model is required)
endif

.require-splits:
ifndef splits
	$(error splits is required)
endif

.require-trials:
ifndef trials
	$(error trials is required)
endif
