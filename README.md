# AI Compiler Study
A repository to study ai compiler

## Run with Docker
This is a simple guideline to run codes in docker container.
### Build a Docker Image
```bash
make docker_image
```
### Run a Docker Container and the Example Code
```bash
make run
```

## Test
You can test the package with following commands.
```bash
# install the package
pip install -e .[testing]

# run tests
pytest
```

## References
- [Triton Github](https://github.com/openai/triton)
- [Triton Webpage](https://triton-lang.org)



## License
MIT
