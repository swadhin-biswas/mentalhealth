# SARAS: Mental Health Support System

SARAS (Supportive AI for Reassurance And Safety) is a sophisticated, retrieval-augmented mental health support system designed to provide empathetic, safe, and context-aware assistance. It leverages a multi-encoder architecture combined with Knowledge Graph integration and a multi-task safety verification system.

## 🌟 Key Features

- **Multi-Encoder Architecture**: Combines specialized clinical embeddings (BioBERT) with general semantic embeddings (MiniLM) for comprehensive text understanding.
- **Knowledge Graph (KG) Integration**: Utilizes a custom-built mental health knowledge graph containing entities, treatments, resources, and coping strategies.
- **RAG-Enhanced Generation**: Employs Retrieval-Augmented Generation to incorporate factual support resources and clinical knowledge into responses.
- **Advanced Safety System**: Includes a dedicated safety verifier with crisis detection, harm classification, and resource prediction heads.
- **Multi-Task Learning**: Optimized for multiple objectives, including crisis detection accuracy, response quality, and clinical relevance.
- **Automated Pipeline**: End-to-end training and evaluation script with comprehensive reporting and logging.

## 🏗️ Architecture Overview

The system is built on a modular architecture:

- **MultiEncoder**: Parallel processing through BioBERT and all-MiniLM-L6-v2 with weighted attention pooling.
- **Hierarchical Attention**: Deep contextual processing using transformer encoder layers.
- **KG Integration Layer**: Fuses entity and relation embeddings from the mental health knowledge graph.
- **Safety Verifier**: A multi-head classifier that monitors for crisis levels, self-harm risk, and the need for emergency resources.
- **Response Quality Head**: Predicts and optimizes for the supportive quality of generated content.

## 📂 Project Structure

```text
/
├── data/               # Reddit mental health data (.jsonl files)
├── src/                # Core implementation
│   ├── model_architecture.py   # Multi-encoder & Safety heads
│   ├── training_pipeline.py    # Training logic & Loss functions
│   ├── data_preprocessing.py   # Reddit data cleaners & Loaders
│   ├── knowledge_graph_rag.py  # KG building & RAG logic
│   ├── inference.py            # Main Assistant interface
│   ├── evaluation_safety.py    # Safety monitoring & Metrics
│   └── report_generator.py     # Automated PDF/JSON reporting
├── models/             # Local pretrained model storage
├── checkpoints/        # Saved model weights
├── logs/               # Training & Execution logs
├── reports/            # Data statistics & Final reports
├── train.sh            # Automated setup & training script
└── requirements.txt    # Project dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (recommended)
- `uv` package manager (optional, used by `train.sh`)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd mental
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR using uv (faster)
   uv pip install -r requirements.txt
   ```

### Training

The project includes an automated training script that handles environment setup, dependency installation, data preprocessing, and training:

```bash
chmod +x train.sh
./train.sh
```

This script will:
- Create a virtual environment (`.venv`).
- Install all required packages.
- Process the Reddit data in `data/`.
- Run the 10-epoch training pipeline.
- Generate a comprehensive report in `reports/`.

### Inference

To use the model for supportive response generation:

```python
from src.inference import MentalHealthAssistant

# Initialize the assistant
assistant = MentalHealthAssistant()
assistant.load_checkpoint("checkpoints/model_best.pt")

# Generate a response
user_input = "I've been feeling really overwhelmed and anxious lately."
response = assistant.generate_response(user_input)

print(f"Response: {response['text']}")
print(f"Safety Status: {response['safety']['status']}")
```

## 🛡️ Safety & Ethics

SARAS is designed with a **Safety-First** philosophy:

1. **Crisis Detection**: Every input is screened for 4 levels of crisis severity.
2. **Harm Filtering**: Automated detection of self-harm or violent intent.
3. **Resource Injection**: If a crisis is detected, the system overrides standard generation to provide verified emergency resources (e.g., 988 Lifeline).
4. **Clinical Grounding**: Responses are grounded in the Knowledge Graph to ensure advice aligns with established coping strategies.

*Note: This system is a supportive tool and not a replacement for professional clinical help.*

## 📊 Documentation

- **Phase 1 Defence Report**: See `report/Phase 1 Defence Report.pdf` for in-depth architectural details and initial findings.
- **Execution Logs**: Detailed JSON logs for every run are stored in `logs/`.
- **Statistics**: Data distributions and class balance reports are available in `reports/`.
