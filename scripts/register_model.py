"""
MLflow Model Registration Script
=================================

This script registers trained models to MLflow Model Registry.

Usage:
    python scripts/register_model.py --model-path models/weights/cicids_moe_best.pt --name moe-cybersecurity-cicids

Features:
    - Register model to MLflow Registry
    - Add metadata (tags, description)
    - Transition to appropriate stage
    - Link to training run
"""

import argparse
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import torch
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.moe_model import load_moe_model
from src.data.feature_config import get_feature_split


def register_model_to_mlflow(
    model_path: str,
    model_name: str,
    dataset: str,
    stage: str = "Staging",
    description: str = None,
    tags: dict = None
):
    """
    Register a trained model to MLflow Model Registry.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        model_name: Name for registered model (e.g., "moe-cybersecurity-cicids")
        dataset: Dataset name (CICIDS or UNSW)
        stage: Initial stage (None, Staging, Production)
        description: Model description
        tags: Dictionary of tags to add
    
    Returns:
        model_version: MLflow ModelVersion object
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 REGISTERING MODEL TO MLFLOW")
    print(f"{'='*80}\n")
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Load model checkpoint
    print(f"📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract metadata from checkpoint
    config = checkpoint.get('config', {})
    metrics = {}
    if 'best_f1' in checkpoint:
        metrics['f1_score'] = checkpoint['best_f1']
    if 'best_precision' in checkpoint:
        metrics['precision'] = checkpoint['best_precision']
    if 'best_recall' in checkpoint:
        metrics['recall'] = checkpoint['best_recall']
    
    print(f"\n📊 Model Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Get feature counts from checkpoint config
    print(f"\n🔢 Getting feature configuration from checkpoint...")
    n_tabular = config.get('n_tabular', 47 if dataset == 'CICIDS' else 185)
    n_temporal = config.get('n_temporal', 25 if dataset == 'CICIDS' else 8)
    print(f"   Tabular features: {n_tabular}")
    print(f"   Temporal features: {n_temporal}")
    
    # Load the actual model
    print(f"\n🔧 Loading MoE model architecture...")
    device = torch.device('cpu')  # Load on CPU for registration
    model = load_moe_model(
        dataset=dataset,
        n_tabular=n_tabular,
        n_temporal=n_temporal,
        model_path=model_path,
        device=device
    )
    model.eval()
    
    # Set experiment
    experiment_name = f"moe-registration-{dataset.lower()}"
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"register_{model_name}") as run:
        run_id = run.info.run_id
        
        print(f"\n📝 MLflow Run ID: {run_id}")
        
        # Log parameters
        print(f"\n📋 Logging parameters...")
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("architecture", "MoE")
        mlflow.log_param("temporal_expert", "1D-CNN")
        mlflow.log_param("tabular_expert", "FT-Transformer")
        
        # Log config parameters
        for key, value in config.items():
            if key != 'feature_names':  # Skip long lists
                mlflow.log_param(f"config_{key}", value)
        
        # Log metrics
        print(f"📊 Logging metrics...")
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model artifact
        print(f"💾 Logging model artifact...")
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            pip_requirements=[
                "torch>=2.0.0",
                "numpy>=1.24.0",
                "pandas>=2.0.0",
                "scikit-learn>=1.3.0"
            ]
        )
        
        print(f"\n✅ Model logged to MLflow!")
    
    # Get the registered model version
    print(f"\n🔍 Finding registered model version...")
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max([int(v.version) for v in versions])
    
    print(f"📦 Model Version: {latest_version}")
    
    # Update model version metadata
    print(f"\n📝 Adding metadata...")
    
    # Add description
    if description is None:
        description = f"MoE model trained on {dataset} dataset. "
        description += f"F1 Score: {metrics.get('f1_score', 0):.4f}. "
        description += "Architecture: FT-Transformer (tabular) + 1D CNN (temporal) + Gating Network."
    
    client.update_model_version(
        name=model_name,
        version=latest_version,
        description=description
    )
    
    # Add tags
    default_tags = {
        "dataset": dataset,
        "architecture": "MoE",
        "temporal_expert": "1D-CNN",
        "tabular_expert": "FT-Transformer",
        "f1_score": f"{metrics.get('f1_score', 0):.4f}"
    }
    
    if tags:
        default_tags.update(tags)
    
    for key, value in default_tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=latest_version,
            key=key,
            value=str(value)
        )
    
    print(f"🏷️  Tags added:")
    for key, value in default_tags.items():
        print(f"   {key}: {value}")
    
    # Transition to stage
    if stage and stage != "None":
        print(f"\n🔄 Transitioning to stage: {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage=stage
        )
    
    print(f"\n{'='*80}")
    print(f"✅ MODEL REGISTRATION COMPLETE!")
    print(f"{'='*80}\n")
    
    print(f"📦 Model Name: {model_name}")
    print(f"📌 Version: {latest_version}")
    print(f"🏷️  Stage: {stage}")
    print(f"🔗 Run ID: {run_id}")
    
    print(f"\n💡 To load this model for inference:")
    print(f"   import mlflow.pyfunc")
    print(f"   model = mlflow.pyfunc.load_model('models:/{model_name}/{stage}')")
    
    print(f"\n💡 To view in MLflow UI:")
    print(f"   mlflow ui --port 5000")
    print(f"   Navigate to: http://localhost:5000")
    
    return latest_version


def list_registered_models():
    """List all registered models in MLflow."""
    
    client = MlflowClient()
    
    print(f"\n{'='*80}")
    print(f"📦 REGISTERED MODELS IN MLFLOW")
    print(f"{'='*80}\n")
    
    models = client.search_registered_models()
    
    if not models:
        print("❌ No registered models found.")
        return
    
    for model in models:
        print(f"📦 {model.name}")
        
        # Get all versions
        versions = client.search_model_versions(f"name='{model.name}'")
        
        if versions:
            print(f"   Versions:")
            for v in sorted(versions, key=lambda x: int(x.version)):
                print(f"      v{v.version}: {v.current_stage}")
                
                # Get metrics from run
                try:
                    run = client.get_run(v.run_id)
                    f1 = run.data.metrics.get('f1_score', 0)
                    print(f"         F1: {f1:.4f}")
                except:
                    pass
        
        print()


def promote_model(model_name: str, version: int, stage: str):
    """
    Promote a model version to a different stage.
    
    Args:
        model_name: Name of registered model
        version: Model version number
        stage: Target stage (Staging, Production, Archived)
    """
    
    client = MlflowClient()
    
    print(f"\n🔄 Promoting model...")
    print(f"   Model: {model_name}")
    print(f"   Version: {version}")
    print(f"   Target Stage: {stage}")
    
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    
    print(f"✅ Model promoted to {stage}!")


def main():
    parser = argparse.ArgumentParser(description="Register model to MLflow Model Registry")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a model')
    register_parser.add_argument('--model-path', type=str, required=True,
                                help='Path to model checkpoint (.pt file)')
    register_parser.add_argument('--name', type=str, required=True,
                                help='Name for registered model')
    register_parser.add_argument('--dataset', type=str, required=True,
                                choices=['CICIDS', 'UNSW'],
                                help='Dataset name')
    register_parser.add_argument('--stage', type=str, default='Staging',
                                choices=['None', 'Staging', 'Production', 'Archived'],
                                help='Initial stage (default: Staging)')
    register_parser.add_argument('--description', type=str, default=None,
                                help='Model description')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all registered models')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote a model to different stage')
    promote_parser.add_argument('--name', type=str, required=True,
                               help='Name of registered model')
    promote_parser.add_argument('--version', type=int, required=True,
                               help='Model version number')
    promote_parser.add_argument('--stage', type=str, required=True,
                               choices=['Staging', 'Production', 'Archived'],
                               help='Target stage')
    
    args = parser.parse_args()
    
    if args.command == 'register':
        register_model_to_mlflow(
            model_path=args.model_path,
            model_name=args.name,
            dataset=args.dataset,
            stage=args.stage,
            description=args.description
        )
    
    elif args.command == 'list':
        list_registered_models()
    
    elif args.command == 'promote':
        promote_model(
            model_name=args.name,
            version=args.version,
            stage=args.stage
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
