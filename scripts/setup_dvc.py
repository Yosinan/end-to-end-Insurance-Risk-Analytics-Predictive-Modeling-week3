"""
Script to set up DVC (Data Version Control) for the project.
This script automates the DVC initialization and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {description} failed")
        print(f"Error output: {e.stderr}")
        return False

def setup_dvc():
    """Set up DVC for the project."""
    print("="*80)
    print("DVC SETUP SCRIPT")
    print("="*80)
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Step 1: Check if DVC is installed
    print("\n1. Checking DVC installation...")
    try:
        result = subprocess.run(['dvc', '--version'], 
                              capture_output=True, text=True)
        print(f"✓ DVC is installed: {result.stdout.strip()}")
    except FileNotFoundError:
        print("✗ DVC is not installed. Installing DVC...")
        if not run_command("pip install dvc", "Installing DVC"):
            print("\n❌ Failed to install DVC. Please install manually:")
            print("   pip install dvc")
            return False
    
    # Step 2: Initialize DVC
    print("\n2. Initializing DVC...")
    if os.path.exists('.dvc'):
        print("⚠ DVC already initialized. Skipping...")
    else:
        if not run_command("dvc init", "Initializing DVC"):
            return False
    
    # Step 3: Create local storage directory
    print("\n3. Setting up local storage...")
    storage_path = project_root / 'dvc_storage'
    storage_path.mkdir(exist_ok=True)
    print(f"✓ Storage directory created: {storage_path}")
    
    # Step 4: Configure local remote storage
    print("\n4. Configuring local remote storage...")
    storage_abs_path = storage_path.absolute()
    
    # Check if remote already exists
    result = subprocess.run(['dvc', 'remote', 'list'], 
                          capture_output=True, text=True)
    if 'localstorage' in result.stdout:
        print("⚠ Remote 'localstorage' already exists. Updating...")
        if not run_command(f"dvc remote modify localstorage url {storage_abs_path}", 
                          "Updating remote storage"):
            return False
    else:
        if not run_command(f"dvc remote add -d localstorage {storage_abs_path}", 
                          "Adding local remote storage"):
            return False
    
    # Step 5: Verify configuration
    print("\n5. Verifying DVC configuration...")
    result = subprocess.run(['dvc', 'remote', 'list'], 
                          capture_output=True, text=True)
    print("Configured remotes:")
    print(result.stdout)
    
    print("\n" + "="*80)
    print("DVC SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Place your data files in data/raw/ directory")
    print("2. Add data files to DVC:")
    print("   dvc add data/raw/insurance_data.csv")
    print("3. Commit DVC files to git:")
    print("   git add data/raw/insurance_data.csv.dvc .gitignore")
    print("   git commit -m 'Add data file with DVC'")
    print("4. Push data to local storage:")
    print("   dvc push")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = setup_dvc()
    sys.exit(0 if success else 1)

