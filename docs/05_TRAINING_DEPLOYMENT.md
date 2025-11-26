# NLP Rules Engine - Training and Deployment Guide

This guide covers fine-tuning the LLM using LoRA and deploying to local Docker or AWS EKS.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Generate Data     2. Train LoRA      3. Export Model        │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐       │
│  │sample_rules  │───>│  Unsloth/    │──>│  GGUF for    │       │
│  │training_data │    │  PEFT        │   │  Ollama      │       │
│  └──────────────┘    └──────────────┘   └──────────────┘       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  4. Deploy                                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Local Docker           OR           AWS EKS              │   │
│  │  (Single machine)                    (Kubernetes)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |
| GPU Type | RTX 3070 | RTX 4090/A100 |

### Software
- NVIDIA CUDA 12.1+
- Docker with GPU support
- Python 3.10+

## Quick Start

### Option 1: All-in-One Script

```bash
# Train and run locally
./training/train_and_deploy.sh --train --local

# Train and deploy to EKS
./training/train_and_deploy.sh --train --deploy \
    --ecr-repo 123456789.dkr.ecr.us-east-1.amazonaws.com \
    --eks-cluster my-cluster
```

### Option 2: Step by Step

```bash
# 1. Generate training data
python training/generate_training_data.py

# 2. Train LoRA model (~2-4 hours)
python training/train_lora.py --backend unsloth --epochs 3

# 3. Export to Ollama format
python training/export_to_ollama.py --output-name rules-llama

# 4. Build Docker image
docker build -f training/Dockerfile.inference -t rules-engine:latest .

# 5. Run locally
docker run --gpus all -p 7860:7860 rules-engine:latest
```

## Training Data

### Generated Format

Training data is in JSONL format (one JSON object per line):

```json
{"messages": [
  {"role": "user", "content": "Create a rule for email validation"},
  {"role": "assistant", "content": "I'll create a REGEX rule..."}
]}
```

### Data Sources

| Source | Description | Examples |
|--------|-------------|----------|
| sample_rules.csv | CSV rule examples | REGEX, NOT_NULL, LENGTH, etc. |
| sample_rules_json.csv | JSON rule examples | Composite, Conditional |
| Clarification | Asking follow-up questions | Missing field prompts |
| Q&A | System knowledge | Rule types, custom functions |
| Typos | Informal language | "gotta have", "aint empty" |

### Customizing Training Data

Add more examples by editing `training/generate_training_data.py`:

```python
# Add custom examples
def generate_custom_examples():
    return [
        {
            "messages": [
                {"role": "user", "content": "Your custom prompt"},
                {"role": "assistant", "content": "Expected response"}
            ]
        }
    ]
```

Then regenerate:
```bash
python training/generate_training_data.py --output training_data.jsonl
```

## Training Configuration

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank (higher = more capacity) |
| `--lora-alpha` | 32 | Scaling factor |
| `--lora-dropout` | 0.05 | Dropout for regularization |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Batch size per GPU |
| `--learning-rate` | 2e-4 | Learning rate |

### Backend Options

```bash
# Unsloth (4x faster, recommended)
python training/train_lora.py --backend unsloth

# HuggingFace PEFT (more compatible)
python training/train_lora.py --backend peft
```

### Memory Optimization

For GPUs with limited VRAM:

```bash
# Reduce batch size
python training/train_lora.py --batch-size 1 --gradient-accumulation 8

# Use smaller model
python training/train_lora.py --model unsloth/llama-3.2-3b-Instruct-bnb-4bit
```

## Docker Training

### Build Training Image

```bash
docker build -f training/Dockerfile.train -t rules-engine-train .
```

### Run Training in Container

```bash
docker run --gpus all \
    -v $(pwd)/training:/app/training \
    -v $(pwd)/sample_rules.csv:/app/sample_rules.csv \
    -v $(pwd)/sample_rules_json.csv:/app/sample_rules_json.csv \
    rules-engine-train
```

### Output Files

After training, find these in `training/output/`:
- `adapter_model.safetensors` - LoRA weights
- `adapter_config.json` - LoRA config
- `training_info.json` - Training metadata

## Model Export

### Export to Ollama

```bash
python training/export_to_ollama.py \
    --model-dir training/output \
    --output-dir training/ollama_export \
    --output-name rules-llama \
    --quantization q4_k_m
```

### Quantization Options

| Option | Size | Speed | Quality |
|--------|------|-------|---------|
| q4_0 | ~4GB | Fastest | Good |
| q4_k_m | ~4.5GB | Fast | Better |
| q5_k_m | ~5.5GB | Medium | Best 5-bit |
| q8_0 | ~8GB | Slower | Near FP16 |
| f16 | ~16GB | Slowest | Full precision |

### Import to Ollama

```bash
cd training/ollama_export
ollama create rules-llama -f Modelfile
ollama run rules-llama
```

## Local Deployment

### Build Inference Image

```bash
docker build -f training/Dockerfile.inference -t rules-engine:latest .
```

### Run with GPU

```bash
docker run -d \
    --name rules-engine \
    --gpus all \
    -p 7860:7860 \
    -p 11434:11434 \
    rules-engine:latest
```

Access at http://localhost:7860

### Run without GPU (CPU only)

```bash
docker run -d \
    --name rules-engine \
    -p 7860:7860 \
    -p 11434:11434 \
    -e OLLAMA_MODEL=llama3.2:3b-instruct-q4_0 \
    rules-engine:latest
```

Note: CPU inference is significantly slower.

## AWS EKS Deployment

### Prerequisites

1. AWS CLI configured
2. EKS cluster with GPU nodes (g4dn.xlarge)
3. NVIDIA device plugin installed
4. ECR repository created

### Push Image to ECR

```bash
# Get ECR login
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin $ECR_REPO

# Tag and push
docker tag rules-engine:latest $ECR_REPO/rules-engine:latest
docker push $ECR_REPO/rules-engine:latest
```

### Deploy to Kubernetes

```bash
# Update kubeconfig
aws eks update-kubeconfig --name my-cluster --region us-east-1

# Deploy
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storage.yaml

# Update deployment with ECR repo
sed "s|\${ECR_REPO}|$ECR_REPO|g" k8s/deployment.yaml | kubectl apply -f -

kubectl apply -f k8s/service.yaml

# Check status
kubectl -n rules-engine get pods
kubectl -n rules-engine get service
```

### EKS Cost Estimate

| Component | Instance | Monthly Cost |
|-----------|----------|--------------|
| GPU Node | g4dn.xlarge | ~$380 |
| Storage | 25GB EBS | ~$2.50 |
| Load Balancer | ALB | ~$20 |
| **Total** | | **~$400/mo** |

Use spot instances for ~60% savings.

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python training/train_lora.py --batch-size 1

# Use smaller model
python training/train_lora.py --model unsloth/llama-3.2-3b-Instruct-bnb-4bit
```

### CUDA Not Found

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Docker GPU Access

```bash
# Install NVIDIA container toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### EKS GPU Node Not Found

```bash
# Check GPU node labels
kubectl get nodes --show-labels | grep gpu

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

## File Reference

```
training/
├── generate_training_data.py  # Create JSONL training data
├── train_lora.py              # LoRA training script
├── export_to_ollama.py        # Export to GGUF/Ollama
├── train_and_deploy.sh        # All-in-one script
├── Dockerfile.train           # Training container
├── Dockerfile.inference       # Inference container
├── requirements.txt           # Python dependencies
├── training_data.jsonl        # Generated training data
├── output/                    # Training output
└── ollama_export/             # Exported model

k8s/
├── namespace.yaml             # Kubernetes namespace
├── deployment.yaml            # Pod deployment
├── service.yaml               # LoadBalancer service
└── storage.yaml               # Persistent volumes
```
