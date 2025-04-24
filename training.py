import torch
import wandb
import os
import logging
import time
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import TPADataset
from mol_former_tpa import MolFormerTPA, ModelEnsemble, predict_tpacs
from constants import (
    BATCH_SIZE, 
    LEARNING_RATE, 
    FROZEN_NUM_EPOCHS, 
    FT_NUM_EPOCHS, 
    DEVICE, 
    USE_MIXED_PRECISION,
    OUTPUT_DIR,
    LOGGING_DIR,
    RANDOM_SEED
)

# Define additional constants
LOGGING_STEPS = 10
EVAL_STEPS = 250
SAVE_STEPS = 250

try:
    os.makedirs(LOGGING_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOGGING_DIR, "training.log")),
            logging.StreamHandler()
        ]
    )
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


def load_data(json_file):
    """Load data from a JSON file"""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File {json_file} not found.")
        return None
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {json_file}.")
        return None


def load_model_dict(file_path):
    """Load a model from a file"""
    try:
        return torch.load(file_path, map_location=DEVICE)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        return None


def get_example_data():
    """Return example data for testing when the real dataset is not available"""
    logger.info("Using example data for testing")
    return [
        {
            "smiles": "CCN(CC)c1ccc2c(c1)O[B-](c1ccccc1)(c1ccccc1)[N+](c1ccccc1)=C2",
            "ET(30)": 37.4,
            "dielectic constant": 7.6,
            "dipole moment": 1.75,
            "wavelength": 790,
            "TPACS": 41,
            "TPACS_log": 1.6127838567197355
        },
        {
            "smiles": "CCN(CC)c1ccc2cc(N(CC)CC)ccc2c1",
            "ET(30)": 45.6,
            "dielectic constant": 32.7,
            "dipole moment": 1.84,
            "wavelength": 800,
            "TPACS": 75,
            "TPACS_log": 1.8750612633917
        },
        {
            "smiles": "CCN(CC)c1ccc2c(c1)SC1=CC=C(N(CC)CC)C=C21",
            "ET(30)": 40.2,
            "dielectic constant": 20.4,
            "dipole moment": 1.92,
            "wavelength": 810,
            "TPACS": 110,
            "TPACS_log": 2.0413926851582249
        }
    ]


def create_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a learning rate scheduler with linear warmup
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        lr_scheduler: Learning rate scheduler
    """
    from transformers import get_scheduler
    
    return get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def train_with_early_stopping(model, train_loader, val_loader, output_dir, 
                             num_epochs, learning_rate, patience=5):
    """
    Train model with early stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Output directory
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        
    Returns:
        str: Path to best model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create optimizer with weight decay for non-bias parameters
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
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=learning_rate)
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = create_lr_scheduler(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_model_path = None
    no_improve_count = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                
                val_loss += outputs["loss"].item()
                all_preds.extend(outputs["predictions"].cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        from sklearn.metrics import r2_score
        r2 = r2_score(np.array(all_labels), np.array(all_preds))
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "r2_score": r2,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # Print metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"R²: {r2:.4f}, "
                    f"Time: {epoch_time:.2f}s")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            
            # Save best model
            best_model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
            
            # Save metrics
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "r2_score": r2,
            }
            with open(os.path.join(output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")
            
            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return best_model_path


def evaluate_model(model, test_loader):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            
            if "loss" in outputs:
                test_loss += outputs["loss"].item()
            all_preds.extend(outputs["predictions"].cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    
    # Calculate average test loss if applicable
    avg_test_loss = test_loss / len(test_loader) if test_loss > 0 else float('nan')
    
    metrics = {
        "test_loss": avg_test_loss,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    return metrics, all_preds, all_labels


def main():
    """Improved main function for training and evaluation"""
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data_file = "TPA_cleaned_data.json"
    all_data = load_data(data_file) or get_example_data()
    
    logger.info(f"Loaded {len(all_data)} data points")

    def get_tpa_bin(item):
        tpa = item.get('TPACS_log', 0)
        if tpa < 1.5:
            return 0
        elif tpa < 2.0:
            return 1
        elif tpa < 2.5:
            return 2
        else:
            return 3
    
    for item in all_data:
        item['tpa_bin'] = get_tpa_bin(item)
    
    from sklearn.model_selection import train_test_split
    
    train_data, temp_data = train_test_split(
        all_data, 
        test_size=0.3, 
        random_state=RANDOM_SEED,
        stratify=[item['tpa_bin'] for item in all_data]
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=RANDOM_SEED,
        stratify=[item['tpa_bin'] for item in temp_data]
    )
    
    logger.info(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    train_dataset = TPADataset(train_data, is_train=True, use_augmentation=True)
    val_dataset = TPADataset(val_data, is_train=False, scaler=train_dataset.scaler)
    test_dataset = TPADataset(test_data, is_train=False, scaler=train_dataset.scaler)

    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=pin_memory
    )
    
    # Initialize
    condition_dim = 4  # wavelength, ET(30), dielectric constant, dipole moment
    model = MolFormerTPA(
        condition_dim=condition_dim, 
        use_mixed_precision=USE_MIXED_PRECISION
    ).to(DEVICE)

    # Report model architecture
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize wandb
    run = wandb.init(
        project="molformer-tpa-prediction",
        name=f"improved-model-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": "MolFormerTPA",
            "dataset_size": len(train_data),
            "augmented_size": len(train_dataset),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "frozen_epochs": FROZEN_NUM_EPOCHS,
            "finetune_epochs": FT_NUM_EPOCHS,
        }
    )

    # Phase 1: Training with frozen MolFormer
    logger.info("Starting Phase 1: Training with frozen MolFormer")
    model.change_molformer(train_molformer=False)
    phase1_output = os.path.join(OUTPUT_DIR, "phase1_improved")
    
    # Train with early stopping
    best_phase1_model_path = train_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase1_output,
        learning_rate=LEARNING_RATE, 
        num_epochs=FROZEN_NUM_EPOCHS,
        patience=5
    )
    
    # Load best model from Phase 1
    if best_phase1_model_path and os.path.exists(best_phase1_model_path):
        model.load_state_dict(torch.load(best_phase1_model_path))
        logger.info(f"Loaded best model from Phase 1: {best_phase1_model_path}")
    else:
        logger.warning("No best model found after Phase 1")

    # Phase 2: Fine-tuning with unfrozen MolFormer
    logger.info("Starting Phase 2: Fine-tuning with unfrozen MolFormer")
    model.change_molformer(train_molformer=True)
    phase2_output = os.path.join(OUTPUT_DIR, "phase2_improved")
    
    # Train with early stopping and reduced learning rate
    best_phase2_model_path = train_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase2_output,
        learning_rate=LEARNING_RATE / 10,  # Reduced learning rate for fine-tuning
        num_epochs=FT_NUM_EPOCHS,
        patience=5
    )
    
    # Load best model from Phase 2
    if best_phase2_model_path and os.path.exists(best_phase2_model_path):
        model.load_state_dict(torch.load(best_phase2_model_path))
        logger.info(f"Loaded best model from Phase 2: {best_phase2_model_path}")
    else:
        logger.warning("No best model found after Phase 2, using last model state")

    # Evaluate on test set
    logger.info("Evaluating model on test set")
    metrics, all_preds, all_labels = evaluate_model(model, test_loader)
    
    # Log test metrics
    for key, value in metrics.items():
        logger.info(f"Test {key}: {value:.4f}")
        wandb.log({f"test_{key}": value})
    
    # Save test predictions
    predictions_file = os.path.join(OUTPUT_DIR, "test_predictions.npz")
    np.savez(
        predictions_file, 
        predictions=all_preds, 
        labels=all_labels,
        metrics=np.array([metrics])
    )
    logger.info(f"Test predictions saved to {predictions_file}")
    
    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Create model ensemble from best checkpoints
    model_paths = []
    
    # Add best model from Phase 1
    if best_phase1_model_path and os.path.exists(best_phase1_model_path):
        model_paths.append(best_phase1_model_path)
    
    # Add best model from Phase 2
    if best_phase2_model_path and os.path.exists(best_phase2_model_path):
        model_paths.append(best_phase2_model_path)
    
    # Add final model
    model_paths.append(final_model_path)
    
    # Create and evaluate ensemble if we have multiple models
    if len(model_paths) > 1:
        logger.info(f"Creating ensemble from {len(model_paths)} models")
        ensemble = ModelEnsemble(MolFormerTPA, model_paths)
        
        # Create a custom evaluator for the ensemble
        def evaluate_ensemble(ensemble, test_loader):
            all_preds = []
            all_uncertainties = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating ensemble"):
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    outputs = ensemble.predict(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        condition=batch['condition']
                    )
                    
                    all_preds.extend(outputs["predictions"].cpu().numpy())
                    all_uncertainties.extend(outputs["uncertainty"].cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            all_preds = np.array(all_preds).flatten()
            all_labels = np.array(all_labels).flatten()
            all_uncertainties = np.array(all_uncertainties).flatten()
            
            mse = mean_squared_error(all_labels, all_preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mean_uncertainty": np.mean(all_uncertainties)
            }
            
            return metrics, all_preds, all_labels, all_uncertainties
        
        ensemble_metrics, ensemble_preds, ensemble_labels, ensemble_uncertainties = evaluate_ensemble(ensemble, test_loader)
        
        logger.info("Ensemble performance:")
        for key, value in ensemble_metrics.items():
            logger.info(f"Ensemble {key}: {value:.4f}")
            wandb.log({f"ensemble_{key}": value})
        
        ensemble_file = os.path.join(OUTPUT_DIR, "ensemble_predictions.npz")
        np.savez(
            ensemble_file, 
            predictions=ensemble_preds, 
            labels=ensemble_labels,
            uncertainties=ensemble_uncertainties,
            metrics=np.array([ensemble_metrics])
        )
        logger.info(f"Ensemble predictions saved to {ensemble_file}")
    
    wandb.finish()
    
    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()