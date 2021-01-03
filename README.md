# aitrace

Research and Trace on Artificial Intelligence.

# Build Docunment

## Requirements

```bash
	# Python3
	pip3 install -r requirements.txt

	# Python2
	pip2 install -r requirements.txt
```




## Build


```bash

cd docs
make html
make pdf

make latex

```


## Git


```bash

git branch gh-pages
git symbolic-ref HEAD refs/heads/gh-pages  # auto-switches branches to gh-pages
rm .git/index
git clean -fdx
git branch

```

