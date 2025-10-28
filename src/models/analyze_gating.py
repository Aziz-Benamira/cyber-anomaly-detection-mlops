"""
Analyze MoE Gating Weights per Attack Type
==========================================
This script loads trained MoE models and analyzes how the gating network
distributes weights between Tabular and Temporal experts for different attack types.

Hypotheses to Test:
- DDoS attacks → Higher temporal weight (timing anomalies)
- PortScan attacks → Higher tabular weight (port/protocol patterns)
- BruteForce attacks → Higher temporal weight (repeated attempts)
- WebAttacks → Balanced weights (payload + timing)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import json
from collections import defaultdict

from src.models.moe_model import load_moe_model
from src.models.train_moe import load_moe_dataset
import yaml


def load_attack_labels(dataset: str):
    """Load original attack labels (before binarization)."""
    raw_data_dir = Path("data/raw")
    
    if dataset == "CICIDS":
        # CICIDS has Label column with attack types
        import pandas as pd
        files = [
            "Monday-WorkingHours.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv", 
            "Wednesday-workingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        ]
        
        labels = []
        for file in files:
            df = pd.read_csv(raw_data_dir / file, encoding='latin1')
            # Clean column names
            df.columns = df.columns.str.strip()
            labels.extend(df['Label'].str.strip().tolist())
        
        return np.array(labels)
    
    elif dataset == "UNSW":
        # UNSW has 'attack_cat' column
        train_df = pd.read_csv(raw_data_dir / "UNSW_NB15_training-set.csv")
        test_df = pd.read_csv(raw_data_dir / "UNSW_NB15_testing-set.csv")
        
        labels = pd.concat([train_df['attack_cat'], test_df['attack_cat']]).values
        return labels
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def extract_gating_weights(model, dataloader, device):
    """Extract gating weights for all samples.
    
    Returns:
        gating_weights: np.array of shape (N, 2) - [tabular_weight, temporal_weight]
    """
    model.eval()
    all_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:  # (X_tabular, X_temporal, y)
                X_tabular, X_temporal, _ = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            
            X_tabular = X_tabular.to(device)
            X_temporal = X_temporal.to(device)
            
            # Forward pass with return_expert_outputs=True to get gating weights
            if X_tabular.dtype == torch.long:
                # UNSW: All categorical features
                _, _, _, gating_weights, _ = model(None, X_tabular, X_temporal, return_expert_outputs=True)
            else:
                # CICIDS: All numerical features
                _, _, _, gating_weights, _ = model(X_tabular, None, X_temporal, return_expert_outputs=True)
            
            all_weights.append(gating_weights.cpu().numpy())
    
    return np.vstack(all_weights)


def analyze_by_attack_type(gating_weights, labels, dataset_name):
    """Analyze gating weights grouped by attack type.
    
    Args:
        gating_weights: np.array (N, 2) - [tabular_weight, temporal_weight]
        labels: np.array (N,) - attack type labels (original multi-class)
        dataset_name: str - "CICIDS" or "UNSW"
    
    Returns:
        results: dict mapping attack_type -> {mean_tabular, mean_temporal, std_tabular, std_temporal, count}
    """
    results = {}
    unique_labels = np.unique(labels)
    
    for attack_type in unique_labels:
        mask = labels == attack_type
        attack_weights = gating_weights[mask]
        
        results[attack_type] = {
            'mean_tabular': attack_weights[:, 0].mean(),
            'mean_temporal': attack_weights[:, 1].mean(),
            'std_tabular': attack_weights[:, 0].std(),
            'std_temporal': attack_weights[:, 1].std(),
            'count': mask.sum(),
            'samples': attack_weights  # For detailed analysis
        }
    
    return results


def visualize_gating_by_attack(results, dataset_name, output_dir):
    """Create visualizations of gating weights per attack type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar plot: Mean gating weights per attack type
    attack_types = list(results.keys())
    mean_tabular = [results[a]['mean_tabular'] for a in attack_types]
    mean_temporal = [results[a]['mean_temporal'] for a in attack_types]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(attack_types))
    width = 0.35
    
    ax.bar(x - width/2, mean_tabular, width, label='Tabular Expert', color='steelblue')
    ax.bar(x + width/2, mean_temporal, width, label='Temporal Expert', color='coral')
    
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Mean Gating Weight', fontsize=12)
    ax.set_title(f'{dataset_name} MoE: Gating Weights per Attack Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_gating_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot: Distribution of gating weights per attack type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Tabular weights
    tabular_data = [results[a]['samples'][:, 0] for a in attack_types]
    bp1 = ax1.boxplot(tabular_data, labels=attack_types, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
    ax1.set_xlabel('Attack Type', fontsize=12)
    ax1.set_ylabel('Tabular Expert Weight', fontsize=12)
    ax1.set_title('Tabular Expert Gating Distribution', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Temporal weights
    temporal_data = [results[a]['samples'][:, 1] for a in attack_types]
    bp2 = ax2.boxplot(temporal_data, labels=attack_types, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('coral')
    ax2.set_xlabel('Attack Type', fontsize=12)
    ax2.set_ylabel('Temporal Expert Weight', fontsize=12)
    ax2.set_title('Temporal Expert Gating Distribution', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_gating_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plot: Tabular vs Temporal weights
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(attack_types)))
    for i, attack_type in enumerate(attack_types):
        samples = results[attack_type]['samples']
        ax.scatter(samples[:, 0], samples[:, 1], 
                  alpha=0.3, s=20, color=colors[i], label=attack_type)
    
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Sum=1 line')
    ax.set_xlabel('Tabular Expert Weight', fontsize=12)
    ax.set_ylabel('Temporal Expert Weight', fontsize=12)
    ax.set_title(f'{dataset_name} MoE: Expert Weight Distribution', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'{dataset_name.lower()}_gating_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualizations to {output_dir}/")


def generate_report(results, dataset_name, output_path):
    """Generate detailed markdown report of gating analysis."""
    report = f"""# MoE Gating Analysis: {dataset_name}

## Overview

This report analyzes how the Mixture-of-Experts (MoE) model distributes weights between the **Tabular Expert** and **Temporal Expert** for different attack types in the {dataset_name} dataset.

## Attack Type Statistics

| Attack Type | Count | Mean Tabular | Mean Temporal | Std Tabular | Std Temporal |
|------------|-------|--------------|---------------|-------------|--------------|
"""
    
    for attack_type, stats in sorted(results.items(), key=lambda x: -x[1]['count']):
        report += f"| {attack_type} | {stats['count']:,} | {stats['mean_tabular']:.4f} | {stats['mean_temporal']:.4f} | {stats['std_tabular']:.4f} | {stats['std_temporal']:.4f} |\n"
    
    report += "\n## Key Findings\n\n"
    
    # Find attack types with highest temporal weights
    sorted_by_temporal = sorted(results.items(), key=lambda x: x[1]['mean_temporal'], reverse=True)
    report += "### Top 3 Attacks Using Temporal Expert:\n\n"
    for i, (attack_type, stats) in enumerate(sorted_by_temporal[:3], 1):
        report += f"{i}. **{attack_type}**: {stats['mean_temporal']:.2%} temporal, {stats['mean_tabular']:.2%} tabular ({stats['count']:,} samples)\n"
    
    # Find attack types with highest tabular weights
    sorted_by_tabular = sorted(results.items(), key=lambda x: x[1]['mean_tabular'], reverse=True)
    report += "\n### Top 3 Attacks Using Tabular Expert:\n\n"
    for i, (attack_type, stats) in enumerate(sorted_by_tabular[:3], 1):
        report += f"{i}. **{attack_type}**: {stats['mean_tabular']:.2%} tabular, {stats['mean_temporal']:.2%} temporal ({stats['count']:,} samples)\n"
    
    # Hypothesis testing
    report += "\n## Hypothesis Validation\n\n"
    
    if dataset_name == "CICIDS":
        # Check DDoS
        ddos_attacks = [k for k in results.keys() if 'DDoS' in k or 'DoS' in k]
        if ddos_attacks:
            ddos_temporal = np.mean([results[k]['mean_temporal'] for k in ddos_attacks])
            report += f"**DDoS Attacks → Temporal Expert?**\n"
            report += f"- Mean temporal weight for DDoS: {ddos_temporal:.2%}\n"
            report += f"- Hypothesis: {'✓ SUPPORTED' if ddos_temporal > 0.5 else '✗ REJECTED'}\n\n"
        
        # Check PortScan
        portscan_attacks = [k for k in results.keys() if 'PortScan' in k or 'Port Scan' in k]
        if portscan_attacks:
            portscan_tabular = np.mean([results[k]['mean_tabular'] for k in portscan_attacks])
            report += f"**PortScan Attacks → Tabular Expert?**\n"
            report += f"- Mean tabular weight for PortScan: {portscan_tabular:.2%}\n"
            report += f"- Hypothesis: {'✓ SUPPORTED' if portscan_tabular > 0.5 else '✗ REJECTED'}\n\n"
        
        # Check Web attacks
        web_attacks = [k for k in results.keys() if 'Web' in k or 'SQL' in k or 'XSS' in k]
        if web_attacks:
            web_tabular = np.mean([results[k]['mean_tabular'] for k in web_attacks])
            web_temporal = np.mean([results[k]['mean_temporal'] for k in web_attacks])
            balance = abs(web_tabular - web_temporal)
            report += f"**Web Attacks → Balanced Experts?**\n"
            report += f"- Mean tabular: {web_tabular:.2%}, Mean temporal: {web_temporal:.2%}\n"
            report += f"- Balance difference: {balance:.2%}\n"
            report += f"- Hypothesis: {'✓ SUPPORTED' if balance < 0.15 else '✗ REJECTED'}\n\n"
    
    elif dataset_name == "UNSW":
        # UNSW attack categories
        report += "**Dataset Note**: UNSW has different attack taxonomy (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms)\n\n"
        
        # Check DoS
        dos_attacks = [k for k in results.keys() if 'DoS' in k or 'dos' in k]
        if dos_attacks:
            dos_temporal = np.mean([results[k]['mean_temporal'] for k in dos_attacks])
            report += f"**DoS Attacks → Temporal Expert?**\n"
            report += f"- Mean temporal weight: {dos_temporal:.2%}\n"
            report += f"- Hypothesis: {'✓ SUPPORTED' if dos_temporal > 0.4 else '✗ REJECTED (Tabular-dominant)'}\n\n"
        
        # Check Reconnaissance (similar to PortScan)
        recon_attacks = [k for k in results.keys() if 'Reconnaissance' in k or 'recon' in k]
        if recon_attacks:
            recon_tabular = np.mean([results[k]['mean_tabular'] for k in recon_attacks])
            report += f"**Reconnaissance Attacks → Tabular Expert?**\n"
            report += f"- Mean tabular weight: {recon_tabular:.2%}\n"
            report += f"- Hypothesis: {'✓ SUPPORTED' if recon_tabular > 0.6 else '✗ REJECTED'}\n\n"
    
    # Overall summary
    overall_tabular = np.mean([stats['mean_tabular'] for stats in results.values()])
    overall_temporal = np.mean([stats['mean_temporal'] for stats in results.values()])
    
    report += f"\n## Overall Summary\n\n"
    report += f"- **Mean Tabular Weight**: {overall_tabular:.2%}\n"
    report += f"- **Mean Temporal Weight**: {overall_temporal:.2%}\n"
    report += f"- **Expert Preference**: {'Tabular-dominant' if overall_tabular > 0.6 else 'Temporal-dominant' if overall_temporal > 0.6 else 'Balanced'}\n"
    report += f"- **Total Samples Analyzed**: {sum(s['count'] for s in results.values()):,}\n"
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved analysis report to {output_path}")


def main():
    """Main analysis pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Analyze MoE gating weights per attack type")
    parser.add_argument('--dataset', type=str, required=True, choices=['CICIDS', 'UNSW'],
                       help='Dataset to analyze')
    parser.add_argument('--output_dir', type=str, default='reports/gating_analysis',
                       help='Output directory for visualizations and reports')
    args = parser.parse_args()
    
    dataset = args.dataset
    output_dir = Path(args.output_dir) / dataset.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING MoE GATING WEIGHTS: {dataset}")
    print(f"{'='*80}\n")
    
    # Load best MoE model
    model_path = Path("models/weights") / f"{dataset.lower()}_moe_best.pt"
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        print("  Please train the MoE model first!")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Loading MoE model from {model_path}")
    
    # Load MoE dataset
    dataset_dir = f"data/processed/{dataset.lower()}"
    train_loader, val_loader, metadata = load_moe_dataset(dataset_dir, batch_size=256)
    
    # Extract feature counts from metadata
    feature_split = metadata.get('feature_split', {})
    n_tabular = feature_split.get('n_tabular', len(feature_split.get('tabular_indices', [])))
    n_temporal = feature_split.get('n_temporal', len(feature_split.get('temporal_indices', [])))
    n_features_total = n_tabular + n_temporal
    
    print(f"[INFO] Loaded dataset: {n_features_total} features")
    print(f"       Tabular: {n_tabular}, Temporal: {n_temporal}")
    
    # Load model
    model = load_moe_model(
        dataset=dataset,
        n_tabular=n_tabular,
        n_temporal=n_temporal,
        model_path=str(model_path),
        device=device
    )
    model.eval()
    
    # Extract gating weights from validation set
    print(f"\n[INFO] Extracting gating weights from validation set...")
    gating_weights = extract_gating_weights(model, val_loader, device)
    print(f"       Extracted {len(gating_weights):,} samples")
    print(f"       Shape: {gating_weights.shape}")
    
    # Load original attack labels
    print(f"\n[INFO] Loading original attack labels...")
    all_labels = load_attack_labels(dataset)
    
    # Need to subsample labels to match validation set size
    # Assuming validation is last 20% of processed dataset
    processed_path = Path("data/processed") / dataset.lower()
    y = np.load(processed_path / "y.npy")
    n_total = len(y)
    n_val = len(gating_weights)
    
    # Get validation indices (last n_val samples)
    val_indices = np.arange(n_total - n_val, n_total)
    
    # Subsample original labels
    if len(all_labels) >= n_total:
        val_labels = all_labels[val_indices]
    else:
        print(f"⚠️  Warning: Label mismatch. Using processed labels instead.")
        val_labels = y[val_indices]
    
    print(f"       Matched {len(val_labels):,} labels")
    print(f"       Unique attack types: {len(np.unique(val_labels))}")
    
    # Analyze gating by attack type
    print(f"\n[INFO] Analyzing gating weights per attack type...")
    results = analyze_by_attack_type(gating_weights, val_labels, dataset)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"GATING ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    
    for attack_type, stats in sorted(results.items(), key=lambda x: -x[1]['count']):
        print(f"{attack_type:30s} (n={stats['count']:6,}): "
              f"Tabular={stats['mean_tabular']:.3f}±{stats['std_tabular']:.3f}, "
              f"Temporal={stats['mean_temporal']:.3f}±{stats['std_temporal']:.3f}")
    
    # Generate visualizations
    print(f"\n[INFO] Creating visualizations...")
    visualize_gating_by_attack(results, dataset, output_dir)
    
    # Generate report
    report_path = output_dir / f"{dataset.lower()}_gating_report.md"
    print(f"\n[INFO] Generating analysis report...")
    generate_report(results, dataset, report_path)
    
    print(f"\n{'='*80}")
    print(f"✅ ANALYSIS COMPLETE!")
    print(f"   Results saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
