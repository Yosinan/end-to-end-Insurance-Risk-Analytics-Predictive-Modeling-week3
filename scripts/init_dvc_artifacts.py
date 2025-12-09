"""
Script to initialize DVC and commit DVC artifacts to git.
This script actually runs dvc init, configures remotes, and commits .dvc files.
"""

import os
import subprocess
import sys
from pathlib import Path

# Add src to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from logger import setup_logger

logger = setup_logger('dvc_init')

def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    logger.info(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            return True, result.stdout
        else:
            logger.warning(f"Command returned non-zero exit code: {result.stderr}")
            return False, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {description} failed - {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Unexpected error during {description}: {str(e)}")
        return False, str(e)

def main():
    """Main function to initialize DVC and commit artifacts."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    logger.info("="*80)
    logger.info("DVC INITIALIZATION AND ARTIFACT COMMIT")
    logger.info("="*80)
    
    # Step 1: Check if DVC is installed
    logger.info("\n1. Checking DVC installation...")
    success, output = run_command("dvc --version", "Checking DVC version", check=False)
    if not success:
        logger.error("DVC is not installed. Please install it first: pip install dvc")
        return False
    logger.info(f"DVC version: {output.strip()}")
    
    # Step 2: Initialize DVC if not already initialized
    logger.info("\n2. Initializing DVC...")
    dvc_dir = project_root / '.dvc'
    if dvc_dir.exists():
        logger.info("DVC already initialized. Skipping...")
    else:
        success, output = run_command("dvc init", "Initializing DVC", check=False)
        if not success:
            logger.error("Failed to initialize DVC")
            return False
    
    # Step 3: Create storage directory
    logger.info("\n3. Creating storage directory...")
    storage_path = project_root / 'dvc_storage'
    try:
        storage_path.mkdir(exist_ok=True)
        logger.info(f"Storage directory: {storage_path}")
    except Exception as e:
        logger.error(f"Error creating storage directory: {str(e)}")
        return False
    
    # Step 4: Configure remote storage
    logger.info("\n4. Configuring remote storage...")
    storage_abs_path = str(storage_path.absolute())
    
    # Check existing remotes
    success, output = run_command("dvc remote list", "Checking existing remotes", check=False)
    if 'localstorage' in output:
        logger.info("Remote 'localstorage' exists. Updating...")
        run_command(f"dvc remote modify localstorage url {storage_abs_path}", 
                   "Updating remote", check=False)
    else:
        success, output = run_command(f"dvc remote add -d localstorage {storage_abs_path}", 
                                     "Adding remote", check=False)
        if not success:
            logger.warning("Failed to add remote, but continuing...")
    
    # Step 5: Verify .dvc/config exists
    logger.info("\n5. Verifying DVC configuration...")
    dvc_config = project_root / '.dvc' / 'config'
    if dvc_config.exists():
        logger.info(f"✓ DVC config file exists: {dvc_config}")
        logger.info("Config file contents:")
        try:
            with open(dvc_config, 'r') as f:
                logger.info(f.read())
        except Exception as e:
            logger.warning(f"Error reading config file: {str(e)}")
    else:
        logger.warning("DVC config file not found")
    
    # Step 6: Add data files to DVC if they exist
    logger.info("\n6. Adding data files to DVC...")
    data_raw_path = project_root / 'data' / 'raw'
    dvc_files_created = []
    
    if data_raw_path.exists():
        data_files = list(data_raw_path.glob('*.csv'))
        for data_file in data_files:
            dvc_file = data_file.with_suffix(data_file.suffix + '.dvc')
            if not dvc_file.exists():
                logger.info(f"Adding {data_file.name} to DVC...")
                success, output = run_command(f"dvc add {data_file}", 
                                             f"Adding {data_file.name}", check=False)
                if success:
                    dvc_files_created.append(dvc_file)
                    logger.info(f"✓ Created {dvc_file}")
                else:
                    logger.warning(f"Failed to add {data_file.name} to DVC")
            else:
                logger.info(f"✓ {data_file.name} already tracked (dvc file exists)")
    else:
        logger.info("data/raw/ directory does not exist. Creating placeholder...")
        data_raw_path.mkdir(parents=True, exist_ok=True)
        # Create a README in data/raw/
        readme_path = data_raw_path / 'README.md'
        readme_path.write_text("# Raw Data Directory\n\nPlace your insurance data CSV files here.\n")
        logger.info("Created data/raw/ directory with README")
    
    # Step 7: Show what should be committed
    logger.info("\n" + "="*80)
    logger.info("DVC SETUP COMPLETE!")
    logger.info("="*80)
    logger.info("\nFiles to commit to git:")
    
    files_to_commit = []
    
    # .dvc/config should be tracked
    if dvc_config.exists():
        files_to_commit.append('.dvc/config')
        logger.info("  - .dvc/config")
    
    # .dvc/.gitignore should be tracked
    dvc_gitignore = project_root / '.dvc' / '.gitignore'
    if dvc_gitignore.exists():
        files_to_commit.append('.dvc/.gitignore')
        logger.info("  - .dvc/.gitignore")
    
    # .dvc files for data should be tracked
    if dvc_files_created:
        for dvc_file in dvc_files_created:
            rel_path = dvc_file.relative_to(project_root)
            files_to_commit.append(str(rel_path))
            logger.info(f"  - {rel_path}")
    
    # Updated .gitignore
    files_to_commit.append('.gitignore')
    logger.info("  - .gitignore")
    
    logger.info("\nTo commit these files, run:")
    logger.info(f"  git add {' '.join(files_to_commit)}")
    logger.info("  git commit -m 'Initialize DVC and add data files'")
    logger.info("\nThen push data to storage:")
    logger.info("  dvc push")
    logger.info("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

