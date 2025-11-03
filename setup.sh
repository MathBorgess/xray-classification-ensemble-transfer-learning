#!/bin/bash

# Setup script for Chest X-Ray Classification Project
# Authors: Jéssica A. L. de Macêdo & Matheus Borges Figueirôa
# CIn - UFPE

echo "=================================================="
echo "Chest X-Ray Classification with Transfer Learning"
echo "=================================================="
echo "Authors:"
echo "  Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)"
echo "  Matheus Borges Figueirôa (mbf3@cin.ufpe.br)"
echo "  CIn - UFPE"
echo "=================================================="

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p results/figures
mkdir -p results/metrics
mkdir -p results/logs

# Create .gitkeep files
touch models/.gitkeep
touch results/.gitkeep

# Test device availability
echo ""
echo "Testing device availability..."
python test_device.py

echo ""
echo "=================================================="
echo "Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Download the Chest X-Ray dataset from Kaggle:"
echo "   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
echo ""
echo "2. Extract the dataset to: data/raw/chest_xray/"
echo ""
echo "3. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Train individual models:"
echo "   python train.py --model efficientnet_b0"
echo "   python train.py --model resnet50"
echo "   python train.py --model densenet121"
echo ""
echo "5. Create and evaluate ensemble:"
echo "   python ensemble.py"
echo ""
echo "6. Test robustness:"
echo "   python test_robustness.py --model efficientnet_b0 --model_path models/efficientnet_b0_final.pth"
echo ""
