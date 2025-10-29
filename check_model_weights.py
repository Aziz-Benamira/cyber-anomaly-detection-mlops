"""Check model weights and training status"""
import torch

model_path = 'models/weights/cicids_moe_best.pt'
print(f"Loading model from: {model_path}")
state = torch.load(model_path, map_location='cpu')

print("\n" + "=" * 70)
print("📊 Model Checkpoint Analysis")
print("=" * 70)
print(f"\nCheckpoint type: {type(state)}")

if isinstance(state, dict):
    print(f"\nTop-level keys: {list(state.keys())}")
    
    # Check training metrics
    if 'best_f1' in state:
        print(f"\n✅ TRAINED MODEL DETECTED!")
        print(f"   Best F1 Score: {state['best_f1']:.4f}")
        print(f"   Best Precision: {state.get('best_precision', 'N/A'):.4f}")
        print(f"   Best Recall: {state.get('best_recall', 'N/A'):.4f}")
        print(f"   Epoch: {state.get('epoch', 'N/A')}")
    else:
        print("\n⚠️  NO TRAINING METRICS FOUND - Model may be untrained!")
        
    # Check model weights
    if 'model_state_dict' in state:
        model_dict = state['model_state_dict']
        print(f"\n📦 Model state dict contains {len(model_dict)} parameter tensors")
        
        # Show first 5 layers
        print("\nFirst 5 layers:")
        for i, key in enumerate(list(model_dict.keys())[:5]):
            tensor = model_dict[key]
            print(f"  {i+1}. {key}")
            print(f"      Shape: {tensor.shape}, Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
        
        # Check if weights look random (untrained) or learned
        first_weight_key = [k for k in model_dict.keys() if 'weight' in k][0]
        first_weight = model_dict[first_weight_key]
        
        print(f"\n🔍 Weight Analysis ({first_weight_key}):")
        print(f"   Mean: {first_weight.mean().item():.8f}")
        print(f"   Std: {first_weight.std().item():.8f}")
        print(f"   Min: {first_weight.min().item():.8f}")
        print(f"   Max: {first_weight.max().item():.8f}")
        
        if abs(first_weight.mean().item()) < 0.0001 and 0.01 < first_weight.std().item() < 0.1:
            print("   ⚠️  Weights look like random initialization (mean≈0, small std)")
        else:
            print("   ✅ Weights appear to be trained (non-zero mean or large std)")
