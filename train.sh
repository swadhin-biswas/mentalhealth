#!/bin/bash

# Mental Health Support Model - Training Script
# Installs dependencies and runs training with JSON logging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
REPORT_DIR="$SCRIPT_DIR/reports"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

# Generate timestamp for this run
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${RUN_ID}.json"
TRAIN_LOG_FILE="$LOG_DIR/${RUN_ID}_training.log"

echo "=========================================="
echo "Mental Health Support Model Training"
echo "=========================================="
echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo ""

# Initialize log JSON
echo '{"run_id": "'"$RUN_ID"'", "status": "starting", "start_time": "'"$(date -Iseconds)"'", "steps": []}' > "$LOG_FILE"

# Function to update JSON log
update_log() {
    local status="$1"
    local message="$2"
    local extra="$3"
    
    python3 -c "
import json

log_file = '$LOG_FILE'

with open(log_file, 'r') as f:
    data = json.load(f)

data['status'] = '$status'
data['message'] = '$message'

$extra

with open(log_file, 'w') as f:
    json.dump(data, f, indent=2)
"
}

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    update_log "installing_uv" "Installing uv package manager" "data['step'] = 'installing_uv'"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "[2/5] Setting up virtual environment and dependencies..."
update_log "installing_dependencies" "Installing Python dependencies" "data['step'] = 'installing_dependencies'"

cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    uv venv .venv
else
    echo "Using existing virtual environment in .venv"
fi

source .venv/bin/activate

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt
else
    echo "Installing default dependencies..."
    uv pip install torch>=2.0.0 \
        numpy>=1.24.0 \
        transformers>=4.30.0 \
        sentence-transformers>=2.2.0 \
        scikit-learn>=1.3.0 \
        pandas>=2.0.0 \
        tqdm>=4.65.0 \
        accelerate>=0.20.0 \
        peft>=0.4.0 \
        scipy>=1.10.0
fi

# Download models (only if not already downloaded locally)
echo "[3/5] Checking pretrained models..."
update_log "downloading_models" "Checking pretrained models" "data['step'] = 'downloading_models'"

$SCRIPT_DIR/.venv/bin/python -c "
import os

models_dir = './models'
mini_model = os.path.join(models_dir, 'all-MiniLM-L6-v2')
biobert_model = os.path.join(models_dir, 'biobert-base-cased-v1.1')

if os.path.exists(mini_model) and os.path.exists(biobert_model):
    print('Models already exist locally, skipping download.')
    print(f'  - {mini_model}')
    print(f'  - {biobert_model}')
else:
    print('ERROR: Models not found in ./models/')
    print('Please run the model download script first.')
    exit(1)
"

# Run data preprocessing
echo "[4/5] Processing data..."
update_log "processing_data" "Processing Reddit mental health data" "data['step'] = 'processing_data'"

$SCRIPT_DIR/.venv/bin/python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$SCRIPT_DIR/src')
from data_preprocessing import MentalHealthDataProcessor
import json

processor = MentalHealthDataProcessor('./data', min_score=2)
dataset = processor.load_all_data()
stats = processor.get_data_statistics(dataset)

print('Data processing complete!')
print(f'Total samples: {stats[\"total_samples\"]}')
print(f'Crisis samples: {stats[\"crisis_samples\"]}')

# Save data stats
with open('$REPORT_DIR/data_stats_${RUN_ID}.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(json.dumps(stats, indent=2))
" 2>&1 | tee -a "$TRAIN_LOG_FILE"

# Run training
echo "[5/5] Starting training..."
update_log "training" "Training model" "data['step'] = 'training'"

export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/src:$PYTHONPATH"

# Use python from venv
PYTHON_CMD="$SCRIPT_DIR/.venv/bin/python"

$PYTHON_CMD -c "
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, '$SCRIPT_DIR')
sys.path.insert(0, '$SCRIPT_DIR/src')
from training_pipeline import setup_training
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

config = {
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'num_epochs': 10,
    'batch_size': 4,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'crisis_weight': 2.0,
    'quality_weight': 0.5,
    'harmony_weight': 0.1,
    'patience': 3,
    'checkpoint_dir': './checkpoints'
}

print(f'Using device: {config[\"device\"]}')
print('Starting training setup...')

trainer, info = setup_training('./data', training_config=config)

print('Training started...')
start_time = time.time()

results = trainer.train()

end_time = time.time()
training_time = end_time - start_time

print(f'Training completed in {training_time:.2f} seconds')

# Get final metrics
final_metrics = trainer.metrics.get_summary()

# Save training results
training_results = {
    'run_id': '$RUN_ID',
    'status': 'completed',
    'end_time': datetime.now().isoformat(),
    'training_time_seconds': training_time,
    'final_metrics': final_metrics,
    'config': config
}

with open('$REPORT_DIR/training_results_${RUN_ID}.json', 'w') as f:
    json.dump(training_results, f, indent=2)

# Update main log
with open('$LOG_FILE', 'r') as f:
    log_data = json.load(f)

log_data['status'] = 'completed'
log_data['end_time'] = datetime.now().isoformat()
log_data['training_time_seconds'] = training_time
log_data['final_metrics'] = final_metrics

with open('$LOG_FILE', 'w') as f:
    json.dump(log_data, f, indent=2)

print('Training results saved!')
print(json.dumps(final_metrics, indent=2))
" 2>&1 | tee -a "$TRAIN_LOG_FILE"

# Generate final report
echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

# Create comprehensive report
$SCRIPT_DIR/.venv/bin/python -c "
import json
import os
from datetime import datetime

run_id = '$RUN_ID'
report_dir = '$REPORT_DIR'
log_dir = '$LOG_DIR'

# Gather all reports
report = {
    'run_id': run_id,
    'generated_at': datetime.now().isoformat(),
    'sections': {}
}

# Data statistics
data_stats_path = f'{report_dir}/data_stats_{run_id}.json'
if os.path.exists(data_stats_path):
    with open(data_stats_path) as f:
        report['sections']['data_statistics'] = json.load(f)

# Training results
training_path = f'{report_dir}/training_results_{run_id}.json'
if os.path.exists(training_path):
    with open(training_path) as f:
        report['sections']['training_results'] = json.load(f)

# Main log
log_path = f'{log_dir}/{run_id}.json'
if os.path.exists(log_path):
    with open(log_path) as f:
        report['sections']['execution_log'] = json.load(f)

# Summary
report['summary'] = {
    'total_samples': report['sections'].get('data_statistics', {}).get('total_samples', 0),
    'crisis_samples': report['sections'].get('data_statistics', {}).get('crisis_samples', 0),
    'training_time': report['sections'].get('training_results', {}).get('training_time_seconds', 0),
    'final_loss': report['sections'].get('training_results', {}).get('final_metrics', {}).get('avg_train_loss', 0),
    'val_loss': report['sections'].get('training_results', {}).get('final_metrics', {}).get('avg_val_loss', 0),
    'crisis_accuracy': report['sections'].get('training_results', {}).get('final_metrics', {}).get('crisis_accuracy', 0)
}

# Save final report
with open(f'{report_dir}/final_report_{run_id}.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Final Report:')
print(json.dumps(report['summary'], indent=2))
print(f'')
print(f'Reports saved to: {report_dir}/')
print(f'Logs saved to: {log_dir}/')
"

echo ""
echo "Log file: $LOG_FILE"
echo "Reports: $REPORT_DIR"
