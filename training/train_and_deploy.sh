#!/bin/bash
# =============================================================================
# NLP Rules Engine - Training and Deployment Script
# =============================================================================
#
# This script handles:
# 1. Local training with GPU
# 2. Model export to Ollama format
# 3. Docker image building
# 4. (Optional) Push to ECR and deploy to EKS
#
# Usage:
#   ./train_and_deploy.sh --train          # Train locally
#   ./train_and_deploy.sh --train --deploy # Train and deploy to EKS
#   ./train_and_deploy.sh --deploy         # Deploy existing model to EKS
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAINING_DIR="${PROJECT_ROOT}/training"
MODEL_NAME="rules-llama"
IMAGE_NAME="rules-engine"
IMAGE_TAG="latest"

# Default options
DO_TRAIN=false
DO_DEPLOY=false
DO_LOCAL=false
ECR_REPO=""
EKS_CLUSTER=""
AWS_REGION="us-east-1"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            DO_TRAIN=true
            shift
            ;;
        --deploy)
            DO_DEPLOY=true
            shift
            ;;
        --local)
            DO_LOCAL=true
            shift
            ;;
        --ecr-repo)
            ECR_REPO="$2"
            shift 2
            ;;
        --eks-cluster)
            EKS_CLUSTER="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --train          Run LoRA training locally"
            echo "  --deploy         Deploy to EKS (requires --ecr-repo)"
            echo "  --local          Run locally with Docker (not EKS)"
            echo "  --ecr-repo URL   ECR repository URL"
            echo "  --eks-cluster N  EKS cluster name"
            echo "  --region REGION  AWS region (default: us-east-1)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}NLP Rules Engine - Train & Deploy${NC}"
echo -e "${GREEN}========================================${NC}"

# =============================================================================
# Step 1: Generate Training Data
# =============================================================================
generate_training_data() {
    echo -e "\n${YELLOW}Step 1: Generating training data...${NC}"

    cd "${PROJECT_ROOT}"
    python training/generate_training_data.py

    echo -e "${GREEN}✓ Training data generated${NC}"
}

# =============================================================================
# Step 2: Train LoRA Model
# =============================================================================
train_model() {
    echo -e "\n${YELLOW}Step 2: Training LoRA model...${NC}"

    # Check for GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}ERROR: NVIDIA GPU not found. Training requires a GPU.${NC}"
        echo "Options:"
        echo "  1. Run on a machine with NVIDIA GPU"
        echo "  2. Use cloud GPU (AWS, GCP, etc.)"
        echo "  3. Use Google Colab with GPU runtime"
        exit 1
    fi

    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv

    cd "${PROJECT_ROOT}"

    # Check if using Docker or local Python
    if command -v docker &> /dev/null && [ -f "${TRAINING_DIR}/Dockerfile.train" ]; then
        echo "Training with Docker..."

        docker build -f training/Dockerfile.train -t rules-engine-train .
        docker run --gpus all \
            -v "${TRAINING_DIR}:/app/training" \
            -v "${PROJECT_ROOT}/sample_rules.csv:/app/sample_rules.csv" \
            -v "${PROJECT_ROOT}/sample_rules_json.csv:/app/sample_rules_json.csv" \
            rules-engine-train
    else
        echo "Training with local Python..."

        # Check dependencies
        python -c "import unsloth" 2>/dev/null || pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        python -c "import peft" 2>/dev/null || pip install peft transformers datasets trl bitsandbytes accelerate

        python training/train_lora.py --backend unsloth --epochs 3
    fi

    echo -e "${GREEN}✓ Training complete${NC}"
}

# =============================================================================
# Step 3: Export Model to Ollama Format
# =============================================================================
export_model() {
    echo -e "\n${YELLOW}Step 3: Exporting model to Ollama format...${NC}"

    cd "${PROJECT_ROOT}"
    python training/export_to_ollama.py \
        --model-dir training/output \
        --output-dir training/ollama_export \
        --output-name "${MODEL_NAME}"

    echo -e "${GREEN}✓ Model exported${NC}"
}

# =============================================================================
# Step 4: Build Docker Image
# =============================================================================
build_docker_image() {
    echo -e "\n${YELLOW}Step 4: Building Docker image...${NC}"

    cd "${PROJECT_ROOT}"

    # Build inference image
    docker build -f training/Dockerfile.inference -t "${IMAGE_NAME}:${IMAGE_TAG}" .

    echo -e "${GREEN}✓ Docker image built: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
}

# =============================================================================
# Step 5: Run Locally
# =============================================================================
run_local() {
    echo -e "\n${YELLOW}Running locally...${NC}"

    docker run -d \
        --name rules-engine \
        --gpus all \
        -p 7860:7860 \
        -p 11434:11434 \
        "${IMAGE_NAME}:${IMAGE_TAG}"

    echo -e "${GREEN}✓ Running at http://localhost:7860${NC}"
    echo "Logs: docker logs -f rules-engine"
}

# =============================================================================
# Step 6: Deploy to EKS
# =============================================================================
deploy_to_eks() {
    echo -e "\n${YELLOW}Step 5: Deploying to EKS...${NC}"

    if [ -z "${ECR_REPO}" ]; then
        echo -e "${RED}ERROR: ECR repository not specified. Use --ecr-repo${NC}"
        exit 1
    fi

    # Login to ECR
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "${AWS_REGION}" | \
        docker login --username AWS --password-stdin "${ECR_REPO}"

    # Tag and push image
    echo "Pushing image to ECR..."
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${ECR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${ECR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

    # Update kubeconfig
    if [ -n "${EKS_CLUSTER}" ]; then
        echo "Updating kubeconfig for cluster: ${EKS_CLUSTER}"
        aws eks update-kubeconfig --name "${EKS_CLUSTER}" --region "${AWS_REGION}"
    fi

    # Deploy to Kubernetes
    echo "Deploying to Kubernetes..."
    cd "${PROJECT_ROOT}"

    # Replace ECR_REPO placeholder in deployment
    sed "s|\${ECR_REPO}|${ECR_REPO}|g" k8s/deployment.yaml > /tmp/deployment.yaml

    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/storage.yaml
    kubectl apply -f /tmp/deployment.yaml
    kubectl apply -f k8s/service.yaml

    # Wait for deployment
    echo "Waiting for deployment..."
    kubectl -n rules-engine rollout status deployment/rules-engine --timeout=300s

    # Get service URL
    echo -e "\n${GREEN}✓ Deployed to EKS!${NC}"
    echo "Getting service URL..."
    kubectl -n rules-engine get service rules-engine

    EXTERNAL_IP=$(kubectl -n rules-engine get service rules-engine -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "pending")
    echo -e "\nAccess the application at: http://${EXTERNAL_IP}"
}

# =============================================================================
# Main
# =============================================================================

if [ "$DO_TRAIN" = true ]; then
    generate_training_data
    train_model
    export_model
    build_docker_image
fi

if [ "$DO_LOCAL" = true ]; then
    if [ "$DO_TRAIN" = false ]; then
        build_docker_image
    fi
    run_local
elif [ "$DO_DEPLOY" = true ]; then
    if [ "$DO_TRAIN" = false ]; then
        build_docker_image
    fi
    deploy_to_eks
fi

if [ "$DO_TRAIN" = false ] && [ "$DO_DEPLOY" = false ] && [ "$DO_LOCAL" = false ]; then
    echo -e "${YELLOW}No action specified. Use --help for usage.${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"
