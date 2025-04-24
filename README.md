# MolFormerTPA

A deep learning model for predicting two-photon absorption (TPA) cross-sections of molecules.

## Overview

MolFormerTPA combines molecular structure information with experimental conditions to predict TPA cross-sections accurately. Built on the IBM MolFormer architecture, it applies an attention-based fusion mechanism to integrate molecular representations with experimental conditions vital to TPA.

## Architecture

The model consists of four main components:

- **Molecular Structure Encoder**: Processes SMILES strings using MolFormer
- **Condition Encoder**: Handles excitation wavelength and three key solvent features: ET(30), dielectric constant, and dipole moment
- **Attention Fusion**: Combines molecular and condition features
- **Prediction Head**: Makes the final TPA cross-section prediction

## Training Methodology

The model employs a two-phase training approach:
1. **Phase 1**: Train with frozen MolFormer weights
2. **Phase 2**: Fine-tune all parameters with reduced learning rate

Early stopping is used during both phases to prevent overfitting.

## Some Important Features

- **SMILES Augmentation**: Increases dataset diversity for better generalization
- **Ensemble Capabilities**: Combines multiple models for improved predictions with uncertainty estimation
- **Custom Pooling**: Multiple strategies for aggregating molecular representations

## Requirements

- PyTorch
- transformers
- RDKit
- scikit-learn
- wandb (optional)