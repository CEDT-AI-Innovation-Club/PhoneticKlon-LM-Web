# PhoneticKlon-LM: AI Thai Poem Generator

## Overview
PhoneticKlon-LM is an AI-powered Thai poem generator that creates traditional กลอนแปด (Eight-syllable verse) poetry. This application was developed as part of the 2110572 NLP SYSTEM course, focusing on generating grammatically correct Thai poems that follow proper rhyming patterns and poetic structures.

## Features
- Generate Thai กลอนแปด poetry from user prompts
- Multiple fine-tuned models to choose from
- Adjustable generation parameters (temperature, top_p, top_k)
- User-friendly web interface built with Streamlit
- Example prompts to help users get started

## Prerequisites
To run this application on your GPU server, you'll need:
- Docker installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit (nvidia-docker)
- Git

## Setup and Deployment

### 1. Clone the Repository
```sh
git clone https://github.com/[username]/thai-poem-generator.git
cd thai-poem-generator
```

### 2. Build the Docker Image
```sh
docker build -t thai-poem-generator .
```

### 3. Run the Container with GPU Support
```sh
docker run --gpus all -p 8501:8501 -v $(pwd)/data:/app/data thai-poem-generator
```

This will:
- Start the Streamlit application on port 8501
- Make it accessible at http://[your-server-ip]:8501
- Mount a local data directory to store generated poems

## Important Note on Data Collection
**All user prompts and generated poems will be saved to a CSV file** (`poem_history_server.csv`). This data will be used to improve the model performance over time. The file will be stored on the server and can be accessed through the mounted volume.

## Models Available
The application includes three different fine-tuned models:
- `nmt-mixed`: Normalized model with mixed training approach
- `nmt_syllable_mixed`: Normalized model with syllable-aware mixed training
- `non-nmt`: Standard model without normalization

## Memory Requirements
Please ensure your GPU has at least 8GB of VRAM to run the models efficiently. For optimal performance, 16GB+ is recommended.

## Troubleshooting
If you encounter any issues:
1. Check GPU availability with `nvidia-smi`
2. Ensure Docker has GPU access
3. Check the Docker logs: `docker logs [container_id]`
4. Make sure port 8501 is not in use by another application

## Project Team
- กัมปนาท ยิ่งเสรี
- พงศกร แก้วใจดี
- ธนธรณ์ ปยะชาติ

This project builds upon the [Klonsuphap-LM](https://medium.com/@kampanatyingseree4704/klonsuphap-lm-%E0%B9%81%E0%B8%95%E0%B9%88%E0%B8%87%E0%B8%81%E0%B8%A5%E0%B8%AD%E0%B8%99%E0%B9%81%E0%B8%9B%E0%B8%94-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-gpt-2-d2baffc80907) work.
