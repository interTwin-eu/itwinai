lock: export-env
	rm -f locks/*
	conda-lock -p linux-64 -p linux-aarch64 -f environment.yml -k explicit --filename-template 'locks/conda-{platform}.lock'

export-env:
	mamba env export --from-history -n itwin-ai  | grep -v "^prefix: " > environment.yml

%: docker/Dockerfile.%
	docker buildx build -t itwin-ai-$@ --load -f $< .

simple: docker/Dockerfile
	docker buildx build -t itwin-ai --load -f $< .
