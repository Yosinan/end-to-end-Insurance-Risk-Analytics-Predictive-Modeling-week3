"""
Script to set up DVC (Data Version Control) for the project.
This script automates the DVC initialization and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

# Add src to path for logger import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from logger import setup_logger

logger = setup_logger('dvc_setup')

def run_command(command, description, check=True):
    """
    Run a shell command and handle errors with logging.
    
    Parameters:
    -----------
    command : str
        Command to run
    description : str
        Description of what the command does
    check : bool
        Whether to raise exception on error
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    logger.info(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✓ {description} completed successfully")
            if result.stdout:
                logger.debug(result.stdout)
            return True
        else:
            logger.warning(f"Command returned non-zero exit code: {result.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        error_msg = f"Error: {description} failed - {e.stderr}"
        logger.error(error_msg)
        if check:
            raise
        return False
    except Exception as e:
        error_msg = f"Unexpected error during {description}: {str(e)}"
        logger.error(error_msg)
        if check:
            raise
        return False

def setup_dvc():
    """
    Set up DVC for the project with comprehensive error handling.
    
    Returns:
    --------
    bool
        True if setup successful, False otherwise
    """
    logger.info("="*80)
    logger.info("DVC SETUP SCRIPT")
    logger.info("="*80)
    
    # Get project root directory
    try:
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        logger.info(f"Project root: {project_root}")
    except Exception as e:
        logger.error(f"Error setting project root: {str(e)}")
        return False
    
    # Step 1: Check if DVC is installed
    logger.info("\n1. Checking DVC installation...")
    try:
        result = subprocess.run(['dvc', '--version'], 
                              capture_output=True, text=True, check=True)
        logger.info(f"✓ DVC is installed: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.warning("DVC is not installed. Attempting to install DVC...")
        if not run_command("pip install dvc", "Installing DVC", check=False):
            logger.error("Failed to install DVC. Please install manually: pip install dvc")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Error checking DVC version: {e.stderr}")
        return False
    
    # Step 2: Initialize DVC
    logger.info("\n2. Initializing DVC...")
    dvc_dir = project_root / '.dvc'
    if dvc_dir.exists():
        logger.warning("DVC already initialized. Skipping initialization...")
    else:
        if not run_command("dvc init", "Initializing DVC", check=False):
            logger.error("Failed to initialize DVC")
            return False
        
        # Verify .dvc directory was created
        if not dvc_dir.exists():
            logger.error("DVC initialization failed: .dvc directory not created")
            return False
        logger.info("✓ DVC initialized successfully")
    
    # Step 3: Create local storage directory
    logger.info("\n3. Setting up local storage...")
    try:
        storage_path = project_root / 'dvc_storage'
        storage_path.mkdir(exist_ok=True)
        logger.info(f"✓ Storage directory created: {storage_path}")
    except Exception as e:
        logger.error(f"Error creating storage directory: {str(e)}")
        return False
    
    # Step 4: Configure local remote storage
    logger.info("\n4. Configuring local remote storage...")
    try:
        storage_abs_path = str(storage_path.absolute())
        
        # Check if remote already exists
        result = subprocess.run(['dvc', 'remote', 'list'], 
                              capture_output=True, text=True)
        if 'localstorage' in result.stdout:
            logger.warning("Remote 'localstorage' already exists. Updating...")
            if not run_command(f"dvc remote modify localstorage url {storage_abs_path}", 
                            "Updating remote storage", check=False):
                logger.warning("Failed to update remote, but continuing...")
        else:
            if not run_command(f"dvc remote add -d localstorage {storage_abs_path}", 
                            "Adding local remote storage", check=False):
                logger.error("Failed to add remote storage")
                return False
    except Exception as e:
        logger.error(f"Error configuring remote storage: {str(e)}")
        return False
    
    # Step 5: Verify configuration
    logger.info("\n5. Verifying DVC configuration...")
    try:
        result = subprocess.run(['dvc', 'remote', 'list'], 
                              capture_output=True, text=True)
        logger.info("Configured remotes:")
        logger.info(result.stdout if result.stdout else "No remotes configured")
        
        # Verify .dvc/config file exists
        dvc_config = project_root / '.dvc' / 'config'
        if dvc_config.exists():
            logger.info(f"✓ DVC config file exists: {dvc_config}")
        else:
            logger.warning("DVC config file not found, but this may be normal")
    except Exception as e:
        logger.warning(f"Error verifying configuration: {str(e)}")
    
    # Step 6: Add data files to DVC if they exist
    logger.info("\n6. Checking for data files to add to DVC...")
    data_raw_path = project_root / 'data' / 'raw'
    if data_raw_path.exists():
        data_files = list(data_raw_path.glob('*.csv'))
        if data_files:
            logger.info(f"Found {len(data_files)} CSV files in data/raw/")
            for data_file in data_files:
                dvc_file = data_file.with_suffix(data_file.suffix + '.dvc')
                if not dvc_file.exists():
                    logger.info(f"Adding {data_file.name} to DVC...")
                    if run_command(f"dvc add {data_file}", f"Adding {data_file.name} to DVC", check=False):
                        logger.info(f"✓ {data_file.name} added to DVC")
                    else:
                        logger.warning(f"Failed to add {data_file.name} to DVC")
                else:
                    logger.info(f"✓ {data_file.name} already tracked by DVC")
        else:
            logger.info("No CSV files found in data/raw/ directory")
    else:
        logger.info("data/raw/ directory does not exist yet")
    
    logger.info("\n" + "="*80)
    logger.info("DVC SETUP COMPLETE!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Place your data files in data/raw/ directory")
    logger.info("2. Add data files to DVC:")
    logger.info("   dvc add data/raw/insurance_data.csv")
    logger.info("3. Commit DVC files to git:")
    logger.info("   git add data/raw/insurance_data.csv.dvc .dvc/config .gitignore")
    logger.info("   git commit -m 'Add data file with DVC'")
    logger.info("4. Push data to local storage:")
    logger.info("   dvc push")
    logger.info("="*80)
    
    return True

if __name__ == "__main__":
    success = setup_dvc()
    sys.exit(0 if success else 1)

