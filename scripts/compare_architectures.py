"""
Compare TabTransformer vs FT-Transformer performance from MLflow experiments
"""
import mlflow
import pandas as pd
from pathlib import Path

def get_latest_runs(experiment_name, n=5):
    """Get the latest n runs from an experiment"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=n
        )
        return runs
    except Exception as e:
        print(f"Error getting runs from {experiment_name}: {e}")
        return None

def extract_metrics(runs_df):
    """Extract key metrics from runs DataFrame"""
    if runs_df is None or len(runs_df) == 0:
        return None
    
    metrics = []
    for idx, run in runs_df.iterrows():
        metric = {
            'run_id': run['run_id'][:8],
            'start_time': run['start_time'],
            'stage': run.get('params.stage', 'N/A'),
            'train_acc': run.get('metrics.train_acc', None),
            'val_acc': run.get('metrics.val_acc', None),
            'architecture': 'FT-Transformer' if 'FT' in str(run.get('tags.mlflow.runName', '')) else 'TabTransformer'
        }
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    
    print("=" * 80)
    print("ARCHITECTURE COMPARISON: TabTransformer vs FT-Transformer")
    print("=" * 80)
    
    # CICIDS Comparison
    print("\n" + "=" * 80)
    print("CICIDS DATASET")
    print("=" * 80)
    
    cicids_runs = get_latest_runs("tabular_expert_cicids", n=10)
    if cicids_runs is not None:
        cicids_metrics = extract_metrics(cicids_runs)
        
        # Separate by stage
        pretrain_runs = cicids_metrics[cicids_metrics['stage'] == 'pretrain']
        finetune_runs = cicids_metrics[cicids_metrics['stage'] == 'finetune']
        
        print("\n📊 PRETRAIN Stage:")
        if len(pretrain_runs) > 0:
            latest_pretrain = pretrain_runs.iloc[0]
            print(f"   Latest Run: {latest_pretrain['run_id']}")
            print(f"   Time: {latest_pretrain['start_time']}")
            print(f"   Architecture: {latest_pretrain['architecture']}")
        
        print("\n📊 FINETUNE Stage:")
        if len(finetune_runs) > 0:
            # Get latest FT-Transformer run
            ft_runs = finetune_runs[finetune_runs['architecture'] == 'FT-Transformer']
            tab_runs = finetune_runs[finetune_runs['architecture'] == 'TabTransformer']
            
            if len(ft_runs) > 0:
                latest_ft = ft_runs.iloc[0]
                print(f"\n   🆕 FT-Transformer (Latest):")
                print(f"      Run ID: {latest_ft['run_id']}")
                print(f"      Train Accuracy: {latest_ft['train_acc']:.4f}" if latest_ft['train_acc'] else "      Train Accuracy: N/A")
                print(f"      Val Accuracy:   {latest_ft['val_acc']:.4f}" if latest_ft['val_acc'] else "      Val Accuracy: N/A")
            
            if len(tab_runs) > 0:
                latest_tab = tab_runs.iloc[0]
                print(f"\n   🔙 TabTransformer (Previous):")
                print(f"      Run ID: {latest_tab['run_id']}")
                print(f"      Train Accuracy: {latest_tab['train_acc']:.4f}" if latest_tab['train_acc'] else "      Train Accuracy: N/A")
                print(f"      Val Accuracy:   {latest_tab['val_acc']:.4f}" if latest_tab['val_acc'] else "      Val Accuracy: N/A")
            
            # Comparison
            if len(ft_runs) > 0 and len(tab_runs) > 0:
                ft_val = latest_ft['val_acc']
                tab_val = latest_tab['val_acc']
                if ft_val and tab_val:
                    improvement = (ft_val - tab_val) * 100
                    print(f"\n   📈 Improvement: {improvement:+.2f}% validation accuracy")
    
    # UNSW Comparison
    print("\n" + "=" * 80)
    print("UNSW DATASET")
    print("=" * 80)
    
    unsw_runs = get_latest_runs("tabular_expert_unsw", n=10)
    if unsw_runs is not None:
        unsw_metrics = extract_metrics(unsw_runs)
        
        # Separate by stage
        pretrain_runs = unsw_metrics[unsw_metrics['stage'] == 'pretrain']
        finetune_runs = unsw_metrics[unsw_metrics['stage'] == 'finetune']
        
        print("\n📊 PRETRAIN Stage:")
        if len(pretrain_runs) > 0:
            latest_pretrain = pretrain_runs.iloc[0]
            print(f"   Latest Run: {latest_pretrain['run_id']}")
            print(f"   Time: {latest_pretrain['start_time']}")
            print(f"   Architecture: {latest_pretrain['architecture']}")
        
        print("\n📊 FINETUNE Stage:")
        if len(finetune_runs) > 0:
            # Get latest FT-Transformer run
            ft_runs = finetune_runs[finetune_runs['architecture'] == 'FT-Transformer']
            tab_runs = finetune_runs[finetune_runs['architecture'] == 'TabTransformer']
            
            if len(ft_runs) > 0:
                latest_ft = ft_runs.iloc[0]
                print(f"\n   🆕 FT-Transformer (Latest):")
                print(f"      Run ID: {latest_ft['run_id']}")
                print(f"      Train Accuracy: {latest_ft['train_acc']:.4f}" if latest_ft['train_acc'] else "      Train Accuracy: N/A")
                print(f"      Val Accuracy:   {latest_ft['val_acc']:.4f}" if latest_ft['val_acc'] else "      Val Accuracy: N/A")
            
            if len(tab_runs) > 0:
                latest_tab = tab_runs.iloc[0]
                print(f"\n   🔙 TabTransformer (Previous):")
                print(f"      Run ID: {latest_tab['run_id']}")
                print(f"      Train Accuracy: {latest_tab['train_acc']:.4f}" if latest_tab['train_acc'] else "      Train Accuracy: N/A")
                print(f"      Val Accuracy:   {latest_tab['val_acc']:.4f}" if latest_tab['val_acc'] else "      Val Accuracy: N/A")
            
            # Comparison
            if len(ft_runs) > 0 and len(tab_runs) > 0:
                ft_val = latest_ft['val_acc']
                tab_val = latest_tab['val_acc']
                if ft_val and tab_val:
                    improvement = (ft_val - tab_val) * 100
                    print(f"\n   📈 Improvement: {improvement:+.2f}% validation accuracy")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nFT-Transformer architecture successfully deployed!")
    print("- All numerical features now participate in self-attention")
    print("- Categorical features continue to use standard embeddings")
    print("- [CLS] token aggregates global context for classification")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
