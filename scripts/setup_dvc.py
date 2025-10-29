"""
DVC Setup Script
================

This script helps you set up DVC for the cyber anomaly detection project.

Steps:
1. Initialize DVC (if not already done)
2. Configure DVC remote storage
3. Track data files with DVC
4. Update dvc.yaml with complete pipeline

Usage:
    python scripts/setup_dvc.py --remote-type local --remote-path D:\dvc-storage
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys


def run_command(cmd, check=True):
    """Run shell command and return output."""
    print(f"   Running: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        check=False
    )
    
    if result.returncode != 0 and check:
        print(f"   ❌ Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    
    return result


def check_dvc_installed():
    """Check if DVC is installed."""
    result = run_command("dvc version", check=False)
    if result.returncode == 0:
        print(f"✅ DVC is installed: {result.stdout.split()[0]}")
        return True
    else:
        print(f"❌ DVC is not installed.")
        print(f"   Install with: pip install dvc")
        return False


def initialize_dvc():
    """Initialize DVC in the project."""
    print(f"\n{'='*80}")
    print(f"📦 INITIALIZING DVC")
    print(f"{'='*80}\n")
    
    # Check if already initialized
    if Path(".dvc").exists():
        print(f"✅ DVC already initialized (.dvc/ directory exists)")
        return True
    
    print(f"🔧 Initializing DVC...")
    result = run_command("dvc init", check=False)
    
    if result.returncode == 0:
        print(f"✅ DVC initialized successfully!")
        
        # Commit DVC initialization
        print(f"\n📝 Committing DVC initialization to Git...")
        run_command("git add .dvc .dvcignore", check=False)
        run_command('git commit -m "Initialize DVC"', check=False)
        
        return True
    else:
        print(f"❌ Failed to initialize DVC")
        return False


def configure_remote(remote_type, remote_path):
    """Configure DVC remote storage."""
    print(f"\n{'='*80}")
    print(f"🌐 CONFIGURING DVC REMOTE STORAGE")
    print(f"{'='*80}\n")
    
    remote_name = "storage"
    
    # Remove existing default remote if any
    run_command("dvc remote remove storage", check=False)
    
    if remote_type == "local":
        print(f"📁 Setting up local remote storage: {remote_path}")
        
        # Create directory if doesn't exist
        Path(remote_path).mkdir(parents=True, exist_ok=True)
        
        # Add remote
        run_command(f'dvc remote add -d {remote_name} "{remote_path}"')
        print(f"✅ Local remote configured!")
    
    elif remote_type == "s3":
        print(f"☁️  Setting up AWS S3 remote: {remote_path}")
        run_command(f'dvc remote add -d {remote_name} {remote_path}')
        print(f"✅ S3 remote configured!")
        print(f"⚠️  Don't forget to set AWS credentials:")
        print(f"   dvc remote modify {remote_name} access_key_id YOUR_KEY")
        print(f"   dvc remote modify {remote_name} secret_access_key YOUR_SECRET")
    
    elif remote_type == "gs":
        print(f"☁️  Setting up Google Cloud Storage remote: {remote_path}")
        run_command(f'dvc remote add -d {remote_name} {remote_path}')
        print(f"✅ GCS remote configured!")
        print(f"⚠️  Don't forget to set GCP credentials:")
        print(f"   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json")
    
    # Commit remote configuration
    print(f"\n📝 Committing remote configuration to Git...")
    run_command("git add .dvc/config", check=False)
    run_command(f'git commit -m "Configure DVC remote: {remote_type}"', check=False)


def track_data_files():
    """Track data files with DVC."""
    print(f"\n{'='*80}")
    print(f"📊 TRACKING DATA FILES WITH DVC")
    print(f"{'='*80}\n")
    
    # Directories to track
    data_dirs = [
        "data/raw",
        "data/processed",
        "models/weights"
    ]
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"\n📁 Tracking: {data_dir}")
            
            # Check if already tracked
            dvc_file = f"{data_dir}.dvc"
            if Path(dvc_file).exists():
                print(f"   ✅ Already tracked ({dvc_file} exists)")
                continue
            
            # Track with DVC
            result = run_command(f'dvc add "{data_dir}"', check=False)
            
            if result.returncode == 0:
                print(f"   ✅ Tracked successfully!")
                
                # Add .dvc file and .gitignore to Git
                run_command(f'git add "{dvc_file}" "{data_dir}/.gitignore"', check=False)
            else:
                print(f"   ❌ Failed to track: {result.stderr}")
        else:
            print(f"\n⚠️  Directory not found: {data_dir}")
    
    # Commit tracked files
    print(f"\n📝 Committing DVC tracking files to Git...")
    run_command('git commit -m "Track data and models with DVC"', check=False)


def push_to_remote():
    """Push tracked files to DVC remote."""
    print(f"\n{'='*80}")
    print(f"☁️  PUSHING DATA TO DVC REMOTE")
    print(f"{'='*80}\n")
    
    print(f"📤 Uploading data to remote storage...")
    result = run_command("dvc push", check=False)
    
    if result.returncode == 0:
        print(f"✅ Data pushed successfully!")
    else:
        print(f"❌ Failed to push data: {result.stderr}")
        print(f"⚠️  Make sure remote storage is configured correctly")


def show_status():
    """Show DVC status."""
    print(f"\n{'='*80}")
    print(f"📊 DVC STATUS")
    print(f"{'='*80}\n")
    
    # DVC status
    print(f"🔍 DVC Pipeline Status:")
    result = run_command("dvc status", check=False)
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("   ✅ Pipeline is up to date!")
    
    # DVC remote list
    print(f"\n🌐 Configured Remotes:")
    result = run_command("dvc remote list", check=False)
    if result.returncode == 0:
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("   ⚠️  No remotes configured")
    
    # DVC tracked files
    print(f"\n📁 Tracked Files:")
    dvc_files = list(Path(".").rglob("*.dvc"))
    if dvc_files:
        for f in dvc_files:
            print(f"   • {f}")
    else:
        print("   ⚠️  No files tracked with DVC")


def main():
    parser = argparse.ArgumentParser(description="Set up DVC for the project")
    
    parser.add_argument('--remote-type', type=str, default='local',
                       choices=['local', 's3', 'gs'],
                       help='Type of remote storage (default: local)')
    parser.add_argument('--remote-path', type=str, default='D:/dvc-storage',
                       help='Path or URL for remote storage (default: D:/dvc-storage)')
    parser.add_argument('--skip-tracking', action='store_true',
                       help='Skip tracking data files (useful if already tracked)')
    parser.add_argument('--skip-push', action='store_true',
                       help='Skip pushing to remote (useful for initial setup)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🚀 DVC SETUP FOR CYBER ANOMALY DETECTION")
    print(f"{'='*80}\n")
    
    # Check if DVC is installed
    if not check_dvc_installed():
        print(f"\n⚠️  Please install DVC first: pip install dvc")
        return
    
    # Step 1: Initialize DVC
    initialize_dvc()
    
    # Step 2: Configure remote storage
    configure_remote(args.remote_type, args.remote_path)
    
    # Step 3: Track data files
    if not args.skip_tracking:
        track_data_files()
    
    # Step 4: Push to remote
    if not args.skip_push:
        push_to_remote()
    
    # Step 5: Show status
    show_status()
    
    print(f"\n{'='*80}")
    print(f"✅ DVC SETUP COMPLETE!")
    print(f"{'='*80}\n")
    
    print(f"📚 Next Steps:")
    print(f"   1. Review tracked files: dvc status")
    print(f"   2. Run pipeline: dvc repro")
    print(f"   3. Push data: dvc push")
    print(f"   4. Pull on another machine: dvc pull")
    
    print(f"\n💡 Useful Commands:")
    print(f"   • dvc status        - Check pipeline status")
    print(f"   • dvc dag           - Show pipeline DAG")
    print(f"   • dvc repro         - Run pipeline")
    print(f"   • dvc push          - Upload data to remote")
    print(f"   • dvc pull          - Download data from remote")
    print(f"   • dvc remote list   - List configured remotes")


if __name__ == "__main__":
    main()
