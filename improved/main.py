import os
import argparse
import logging
import json
import torch
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import improved modules
from dataset import TPADataset
from mol_former_tpa import MolFormerTPA, predict_tpacs
from training import (
    train_with_early_stopping, 
    evaluate_model, 
    load_data, 
    get_example_data
)

# Import constants
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

# Set up logging
os.makedirs(LOGGING_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGGING_DIR, "main.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TPA Prediction with MolFormer")
    parser.add_argument("--data_path", type=str, default="TPA_cleaned_data.json", 
                        help="Path to the TPA data JSON file")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--phase1_epochs", type=int, default=FROZEN_NUM_EPOCHS,
                        help="Number of epochs for phase 1 (frozen backbone)")
    parser.add_argument("--phase2_epochs", type=int, default=FT_NUM_EPOCHS,
                        help="Number of epochs for phase 2 (fine-tuning)")
    parser.add_argument("--use_augmentation", action="store_true",
                        help="Use SMILES augmentation during training")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run only evaluation on test set using a saved model")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to saved model (for eval_only mode)")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    data = load_data(args.data_path) or get_example_data()
    logger.info(f"Loaded {len(data)} data points")
    
    # Initialize Weights & Biases if enabled
    if not args.no_wandb:
        wandb.init(
            project="tpa-prediction",
            config=vars(args)
        )
    
    # Stratified split
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
    
    # Add bin information
    for item in data:
        item['tpa_bin'] = get_tpa_bin(item)
    
    # Split data
    train_data, temp_data = train_test_split(
        data, 
        test_size=0.3, 
        random_state=RANDOM_SEED,
        stratify=[item['tpa_bin'] for item in data]
    )
    
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=0.5, 
        random_state=RANDOM_SEED,
        stratify=[item['tpa_bin'] for item in temp_data]
    )
    
    logger.info(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = TPADataset(
        train_data, 
        is_train=True, 
        use_augmentation=args.use_augmentation
    )
    
    val_dataset = TPADataset(
        val_data, 
        is_train=False, 
        scaler=train_dataset.scaler
    )
    
    test_dataset = TPADataset(
        test_data, 
        is_train=False, 
        scaler=train_dataset.scaler
    )
    
    # Create data loaders
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Use 0 to avoid multiprocessing issues with tokenizers
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 to avoid multiprocessing issues with tokenizers
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 to avoid multiprocessing issues with tokenizers
        pin_memory=pin_memory
    )
    
    # Initialize model
    condition_dim = 4  # wavelength, ET(30), dielectric constant, dipole moment
    model = MolFormerTPA(
        condition_dim=condition_dim, 
        use_mixed_precision=USE_MIXED_PRECISION
    ).to(DEVICE)
    
    # Eval only mode
    if args.eval_only:
        if args.model_path is None:
            logger.error("Model path must be provided in eval_only mode")
            return
        
        logger.info(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        
        logger.info("Evaluating model on test set")
        metrics, predictions, labels = evaluate_model(model, test_loader)
        
        for key, value in metrics.items():
            logger.info(f"Test {key}: {value:.4f}")
            if not args.no_wandb:
                wandb.log({f"test_{key}": value})
        
        # Save predictions
        import numpy as np
        predictions_file = os.path.join(args.output_dir, "test_predictions.npz")
        np.savez(
            predictions_file, 
            predictions=predictions, 
            labels=labels,
            metrics=np.array([metrics])
        )
        logger.info(f"Test predictions saved to {predictions_file}")
        
        if not args.no_wandb:
            wandb.finish()
        
        return
    
    # Training mode
    # Phase 1: Training with frozen MolFormer
    logger.info("Starting Phase 1: Training with frozen MolFormer")
    model.change_molformer(train_molformer=False)
    phase1_output = os.path.join(args.output_dir, "phase1")
    
    best_phase1_model_path = train_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase1_output,
        learning_rate=args.learning_rate, 
        num_epochs=args.phase1_epochs,
        patience=args.patience
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
    phase2_output = os.path.join(args.output_dir, "phase2")
    
    best_phase2_model_path = train_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        output_dir=phase2_output,
        learning_rate=args.learning_rate / 10,  # Reduced learning rate for fine-tuning
        num_epochs=args.phase2_epochs,
        patience=args.patience
    )
    
    # Load best model from Phase 2
    if best_phase2_model_path and os.path.exists(best_phase2_model_path):
        model.load_state_dict(torch.load(best_phase2_model_path))
        logger.info(f"Loaded best model from Phase 2: {best_phase2_model_path}")
    else:
        logger.warning("No best model found after Phase 2, using last model state")
    
    # Evaluate on test set
    logger.info("Evaluating model on test set")
    metrics, predictions, labels = evaluate_model(model, test_loader)
    
    for key, value in metrics.items():
        logger.info(f"Test {key}: {value:.4f}")
        if not args.no_wandb:
            wandb.log({f"test_{key}": value})
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Save predictions
    import numpy as np
    predictions_file = os.path.join(args.output_dir, "test_predictions.npz")
    np.savez(
        predictions_file, 
        predictions=predictions, 
        labels=labels,
        metrics=np.array([metrics])
    )
    logger.info(f"Test predictions saved to {predictions_file}")
    
    # Sample prediction
    logger.info("Running sample prediction")
    sample_smiles = test_data[0]["smiles"]
    sample_wavelength = test_data[0]["wavelength"]
    sample_et30 = test_data[0]["ET(30)"]
    sample_dielectric = test_data[0]["dielectic constant"]
    sample_dipole = test_data[0]["dipole moment"]
    
    sample_pred = predict_tpacs(
        model,
        [sample_smiles],
        sample_wavelength,
        sample_et30,
        sample_dielectric,
        sample_dipole,
        scaler=train_dataset.scaler
    )
    
    logger.info(f"Sample SMILES: {sample_smiles}")
    logger.info(f"Sample conditions: wavelength={sample_wavelength}, ET(30)={sample_et30}, " +
                f"dielectric={sample_dielectric}, dipole={sample_dipole}")
    logger.info(f"Predicted TPA (log): {sample_pred[0][0]:.4f}")
    logger.info(f"Actual TPA (log): {test_data[0]['TPACS_log']:.4f}")
    
    if not args.no_wandb:
        wandb.finish()
    
    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()