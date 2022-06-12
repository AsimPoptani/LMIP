# Phony
.PHONY: database venv help all update-requirements
all: venv

# Show this help.
help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

database: venv ## Gets the databases for transform.
	@echo "Building database..."
	source venv/bin/activate  && python3 src/DownloadData.py

transform: venv ## Transforms the database.
	@echo "Transforming database..."
	source venv/bin/activate  && python3 src/TransformData.py

venv: ## Create venv and install dependencies.
	test  -d venv || (python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt)

downloadAndTransform: database transform
	@echo "Downloading and transforming data..."

requirements: venv ## Update the project's packages requirements.
	@echo "Updating packages..."
	source venv/bin/activate  && pip freeze > requirements.txt



	

