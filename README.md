## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed on your machine:

- Docker
- Docker Compose

#### GPU Support (Optional)

If you have a GPU and want to leverage its power within a Docker container, follow these steps to install the NVIDIA Container Toolkit:

```bash
./setup_cuda.sh
```

## Usage

Start Ollama and its dependencies using Docker Compose:

if gpu is configured

```bash
docker-compose -f docker-compose-ollama-gpu.yaml up
```

else

```bash
docker-compose up
```

Visit [http://localhost:8000](http://localhost:8000) in your browser.
