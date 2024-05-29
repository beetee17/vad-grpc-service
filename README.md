# vad-grpc-service

This is a repository to serve a Voice Activity Detection (VAD) Service via FastAPI. The model being served is a pre-trained enterprise-grade Voice Activity Detector known as Silero[https://github.com/snakers4/silero-vad/].

## Setup

### 1. Git clone the repository

```sh
git clone https://github.com/beetee17/vad-grpc-service.git
```

### 2. Download the weights

The necessary model files should already be included in the repo as it is quite lightweight. However, if you need to download it manually, you can run the following script:

```py
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=True
)
```

Please place the contents of the download into the `model` folder. The directory should look like:

```sh
vad-grpc-service
|-- model
|    |-- files
|        |-- silero_vad.jit
|        |-- silero_vad.onnx
|    |-- hubconf.py
|    |-- utils.vad.py
|    ...
|-- proto
|-- src
|-- tests
| ...
...
```

### 3. ENV
Set-up the .env file (`main.env`) using the template at `.env_template`. There are 2 main environment variables that to be in:

```sh
GRPC_PORT="XXXXX" # the port to be exposed and used for the service
MODEL_DIR="/model" # path to where the model files are (should not need to be changed)
```

### 4. Docker

Build the dockerfile:
```sh
docker build -t beetee/vad-service:1.0.0-build -f dockerfile.dev --target build .
```

Start up the server:
```sh
docker-compose -f docker-compose.dev.yaml up --build
```

### 5. Sample Requests
You will need to set-up the protobufs. with the following command:

```sh
python -m venv ~/grpc_env
source ~/grpc_env/bin/activate

pip install grpc-tools==1.62.1

python -m grpc_tools.protoc -I . --python_out=./tests --pyi_out=./tests --grpc_python_out=./tests ./proto/vad.proto
python -m grpc_tools.protoc -I . --python_out=./src --pyi_out=./src --grpc_python_out=./src ./proto/vad.proto
```

While the server is up, try streaming requests via the microphone with `tests/stream_request.py`. 

```sh
python -m venv ~/vad_client_env
source ~/vad_client_env/bin/activate
pip install -r client_requirements.txt
cd tests
python stream_request.py
```

#### Note

You may need to install additional dependencies to run the test script.

Make sure to verify the hostname and port number in the python script match that of the docker container:

```sh
docker ps
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_id_or_name>
```