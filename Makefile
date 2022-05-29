SHELL:=/bin/bash
CONTAINER=kmeans_homemade
DIR=kmeans_homemade
jupyterlab:
	docker exec -it ${CONTAINER} bash -c "jupyter-lab --ip 0.0.0.0 --no-browser --port 8888 --allow-root --notebook-dir=/usr/src/${DIR}"
