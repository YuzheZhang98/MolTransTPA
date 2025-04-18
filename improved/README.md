# ImprovedMolFormerTPA: Advanced Two-Photon Absorption Prediction

## Technical Documentation

This document provides in-depth technical information about the ImprovedMolFormerTPA system for predicting two-photon absorption (TPA) cross-sections of molecules under specific experimental conditions.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Components](#model-components)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Training Methodology](#training-methodology)
5. [Overfitting Prevention](#overfitting-prevention)
6. [Performance Metrics](#performance-metrics)
7. [Implementation Details](#implementation-details)
8. [Advanced Usage](#advanced-usage)

## Architecture Overview

ImprovedMolFormerTPA is a multi-modal deep learning system that combines molecular structure information with experimental conditions to predict TPA cross-sections. The architecture consists of four main components:

1. **Molecular Structure Encoder**: Uses the MolFormer transformer-based architecture to convert SMILES strings into rich molecular representations
2. **Experimental Condition Encoder**: Processes wavelength and solvent properties through a specialized MLP
3. **Self-Attention Fusion Module**: Intelligently combines molecular and condition features
4. **Regression Head**: Makes the final TPA prediction based on the fused representations

The system employs a two-phase training strategy with a focus on transfer learning from the pre-trained MolFormer weights.

## Model Components

### Molecular Structure Encoder

The molecular encoder leverages IBM's MolFormer model, which is a transformer-based architecture pre-trained on millions of molecules. Our implementation extends the standard approach by:

- **Advanced Pooling Strategies**: Instead of just using the CLS token, we implement multiple pooling strategies:
  - Mean pooling across all token embeddings (weighted by attention mask)
  - Max pooling to capture the most salient features
  - Attention-based pooling with learned attention weights
  
- **Implementation Details**: The pooling method can be selected via the `pool_type` parameter in the model initialization:

```python
# In the ImprovedMolFormerTPA constructor
self.pool_type = "mean"  # Options: "cls", "mean", "max", "attention"
```

### Experimental Condition Encoder

The condition encoder processes numerical experimental parameters (wavelength, ET(30), dielectric constant, dipole moment) through a multilayer perceptron with:

- Layer normalization for training stability
- Graduated dimensionality (expands then contracts)
- Dropout layers to prevent overfitting
- ReLU activation functions

```python
self.mlp = nn.Sequential(
    nn.Linear(condition_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, hidden_dim * 2),
    nn.LayerNorm(hidden_dim * 2),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim * 2, output_dim)
)
```

### Self-Attention Fusion Module

One of the key features is the self-attention fusion mechanism that effectively combines molecular and condition features:

1. Projects both features to a common dimensionality
2. Applies self-attention to allow each feature to attend to both itself and the other feature
3. Uses residual connections and layer normalization to stabilize training

This approach allows the model to dynamically decide how to weight molecular vs. experimental information for each prediction.

```python
# Project features to common space
mol_feat = self.mol_proj(mol_repr)
cond_feat = self.cond_proj(cond_repr)

# Concatenate features for attention
combined = torch.stack([mol_feat, cond_feat], dim=1)  # [batch, 2, dim]

# Self-attention
q = self.query(combined)
k = self.key(combined)
v = self.value(combined)

# Compute attention scores
scores = torch.matmul(q, k.transpose(-2, -1)) / (combined.size(-1) ** 0.5)
attn_weights = F.softmax(scores, dim=-1)

# Apply attention
attn_output = torch.matmul(attn_weights, v)
```

### Regression Head

The prediction head performs the final regression using a deep MLP with:

- Decreasing dimensionality (512 → 256 → 128 → 64 → 1)
- Layer normalization after each linear layer
- Dropout with progressively decreasing rates (0.3 → 0.2 → 0)
- ReLU activations

## Data Processing Pipeline

### Data Validation

All input data undergoes strict validation to ensure:
- SMILES strings are present and valid
- All numerical parameters are finite and within reasonable ranges
- No missing values in critical fields

Invalid entries are filtered out and reported for inspection.

### SMILES Augmentation

We implement SMILES augmentation as a critical data preprocessing step:

1. Parse each SMILES string with RDKit
2. Generate multiple valid but different SMILES strings for the same molecule
3. Include these augmented representations in training with the same target value

This method effectively increases dataset size and introduces beneficial noise that improves generalization. Since each augmented SMILES represents the exact same molecule, this does not leak information between train/test splits.

```python
def smiles_augmentation(smiles_list, num_augmentations=2):
    augmented_data = {}
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                augmented_data[smiles] = [smiles]  # Keep original if parsing fails
                continue
                
            augmented = [smiles]  # Always include the original
            for _ in range(num_augmentations):
                # Generate random SMILES for the same molecule
                rand_smiles = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                if rand_smiles != smiles and rand_smiles not in augmented:
                    augmented.append(rand_smiles)
            
            augmented_data[smiles] = augmented
        except Exception as e:
            print(f"Error augmenting SMILES {smiles}: {str(e)}")
            augmented_data[smiles] = [smiles]  # Fallback to original
    return augmented_data
```

### Condition Scaling

Experimental conditions are standardized using scikit-learn's StandardScaler to normalize the values to zero mean and unit variance. The scaler is fitted on the training data and applied to validation and test sets to prevent data leakage.

## Training Methodology

### Two-Phase Training

The training process follows a two-phase approach:

#### Phase 1: Feature Extraction
- MolFormer weights are frozen
- Only fusion and prediction layers are trained
- Higher initial learning rate (typically 1e-4)
- Focus on adapting to the specific TPA prediction task

#### Phase 2: Fine-Tuning
- MolFormer weights are unfrozen
- Full model trained with reduced learning rate (typically 1e-5)
- Careful optimization to avoid catastrophic forgetting

### Learning Rate Schedule

A linear learning rate schedule with warm-up is employed:
- 10% of total training steps used for warm-up
- Linear decay after warm-up phase
- Implemented using the transformers library's scheduler

### Loss Function

The model uses a composite loss function combining:
- Mean Squared Error (MSE) as the primary loss
- Mean Absolute Error (L1) as a secondary loss with 0.1 weight

```python
# MSE loss with additional L1 regularization
mse_loss = F.mse_loss(tpa_prediction, labels)
l1_loss = F.l1_loss(tpa_prediction, labels)
loss = mse_loss + 0.1 * l1_loss
```

This combination provides robustness to outliers while maintaining focus on the squared error.

## Overfitting Prevention

Multiple techniques are employed to prevent overfitting:

### 1. Data Augmentation
SMILES augmentation effectively increases the dataset size and variability, exposing the model to different representations of the same molecules.

### 2. Regularization Layers
Dropout layers are strategically placed throughout the network:
- 0.1 dropout in condition encoder
- 0.3 dropout in the first layer of prediction head
- 0.2 dropout in the second layer of prediction head

### 3. Layer Normalization
Layer normalization is applied after every linear layer to stabilize training and reduce internal covariate shift.

### 4. Early Stopping
Training monitors validation loss and stops when it fails to improve for a specified number of epochs (default: 5).

```python
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    no_improve_count = 0
    # Save best model...
else:
    no_improve_count += 1
    logger.info(f"No improvement for {no_improve_count} epochs")
    
    # Early stopping
    if no_improve_count >= patience:
        logger.info(f"Early stopping triggered after {epoch+1} epochs")
        break
```

### 5. Weight Decay
L2 regularization is applied to all non-bias and non-normalization weights with a factor of 0.01.

```python
# Optimizer with weight decay for non-bias parameters
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_params = [
    {
        "params": [p for n, p in model.named_parameters() 
                  if not any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters() 
                  if any(nd in n for nd in no_decay) and p.requires_grad],
        "weight_decay": 0.0,
    },
]
```

### 6. Gradient Clipping
Gradient norms are clipped at 1.0 to prevent exploding gradients.

```python
# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7. Model Ensembling
Multiple models or checkpoints can be ensembled during inference, reducing variance and improving prediction stability.

## Performance Metrics

The model is evaluated using multiple metrics:

1. **Mean Squared Error (MSE)**: Primary metric for regression performance
2. **Root Mean Squared Error (RMSE)**: Provides error in the same units as the target
3. **Mean Absolute Error (MAE)**: Less sensitive to outliers
4. **R² Score**: Indicates the proportion of variance explained by the model
5. **Uncertainty Estimation**: When using model ensembles, prediction standard deviation provides uncertainty estimates

## Implementation Details

### Code Organization

The codebase is organized into modular components:

1. **constants.py**: Central configuration of hyperparameters
2. **dataset.py**: Dataset handling, validation, and augmentation
3. **encoder.py**: Encoder and fusion components
4. **mol_former_tpa.py**: Main model architecture
5. **training.py**: Training loops and evaluation
6. **main.py**: Command-line interface and experiment runner

### Dependencies

The implementation requires:
- PyTorch (>= 1.10.0)
- transformers (>= 4.20.0)
- RDKit (for SMILES manipulation)
- scikit-learn (for data splitting and metrics)
- wandb (optional, for experiment tracking)

### Mixed Precision Training

Mixed precision training can be used on compatible CUDA devices to accelerate training while maintaining numerical stability.

```python
with autocast(device_type=DEVICE.type, enabled=self.use_mixed_precision):
    molformer_output = self.molformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
```

## Advanced Usage

### Model Ensembling

The `ModelEnsemble` class allows combining multiple trained models:

```python
# Load ensemble from multiple checkpoints
ensemble = ModelEnsemble(
    ImprovedMolFormerTPA, 
    ["path/to/model1.pt", "path/to/model2.pt", "path/to/model3.pt"],
    device=torch.device("cuda")
)

# Get predictions with uncertainty
outputs = ensemble.predict(input_ids, attention_mask, condition)
predictions = outputs["predictions"]
uncertainties = outputs["uncertainty"]
```

### Hyperparameter Optimization

For advanced users, the key hyperparameters to optimize include:

- Learning rates for both phases
- Hidden dimensions for encoders
- Dropout rates
- Pooling strategy for molecular representations
- Weight of L1 loss component

### Batch Prediction

For high-throughput virtual screening, the `predict_tpacs` function provides efficient batch prediction:

```python
predictions = predict_tpacs(
    model,
    smiles_list,  # Can be thousands of SMILES
    wavelength,
    et30,
    dielectric,
    dipole,
    batch_size=64,  # Adjust based on memory
    scaler=dataset.scaler
)
```

### Custom Condition Encoding

The model architecture can be extended to incorporate additional experimental conditions or molecular descriptors by modifying the `condition_dim` parameter and preprocessing pipeline.
