.PHONY: build
registry = registry.cn-zhangjiakou.aliyuncs.com/
name = humanseg
version = beta-$(shell git rev-parse --short HEAD)-$(shell git branch --show-current)

release: *.py
	docker build -t ${registry}vikings/paddle:${name}-${version} .
	docker tag ${registry}vikings/paddle:${name}-${version} ${registry}vikings/paddle:custom-${name}
	docker push ${registry}vikings/paddle:${name}-${version}
	docker push ${registry}vikings/paddle:custom-${name}
