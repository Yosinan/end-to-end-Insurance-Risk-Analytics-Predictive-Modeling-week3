# DVC Setup Guide

This guide walks you through setting up Data Version Control (DVC) for the Insurance Risk Analytics project.

## Why DVC?

In finance and insurance, we must be able to reproduce any analysis or model result at any time for auditing, regulatory compliance, or debugging. DVC ensures our data inputs are as rigorously version-controlled as our code.

## Prerequisites

- Python 3.8+
- Git initialized in the project
- Data files ready to be tracked

## Setup Steps

### Step 1: Install DVC

```bash
pip install dvc
```

Or if using the project requirements:

```bash
pip install -r requirements.txt
```

### Step 2: Initialize DVC

```bash
dvc init
```

This creates a `.dvc` directory with DVC configuration files.

### Step 3: Create Local Storage Directory

```bash
mkdir dvc_storage
```

### Step 4: Configure Local Remote Storage

```bash
# Add local remote storage (replace with absolute path)
dvc remote add -d localstorage /path/to/your/project/dvc_storage

# Or use relative path (from project root)
dvc remote add -d localstorage ./dvc_storage
```

Verify the configuration:

```bash
dvc remote list
```

### Step 5: Add Data Files to DVC

Place your data files in the `data/raw/` directory, then add them to DVC:

```bash
# Add a single file
dvc add data/raw/insurance_data.csv

# Add multiple files
dvc add data/raw/*.csv
```

This will:
- Create a `.dvc` file (metadata about your data)
- Add the data file to `.gitignore`
- Store the actual data in DVC cache

### Step 6: Commit Changes to Git

```bash
# Stage the .dvc files and updated .gitignore
git add data/raw/*.dvc .gitignore

# Commit
git commit -m "Add data files with DVC tracking"
```

**Important:** Only commit the `.dvc` files to git, not the actual data files. The data files are tracked by DVC and stored in the remote storage.

### Step 7: Push Data to Local Remote Storage

```bash
dvc push
```

This uploads your data files to the configured remote storage.

## Using DVC

### Retrieving Data

When you clone the repository or switch branches, retrieve data files:

```bash
dvc pull
```

### Checking Status

```bash
# Check if data files have changed
dvc status

# List all tracked files
dvc list .
```

### Comparing Versions

```bash
# Compare current data with previous version
dvc diff
```

### Updating Data

If you update a data file:

```bash
# Re-add the file
dvc add data/raw/insurance_data.csv

# Commit the changes
git add data/raw/insurance_data.csv.dvc
git commit -m "Update data file"

# Push to remote
dvc push
```

## Automated Setup

You can use the provided script to automate the setup:

```bash
python scripts/setup_dvc.py
```

Or use the Jupyter notebook:

```bash
jupyter notebook notebooks/task2_dvc_setup.ipynb
```

## Project Structure After DVC Setup

```
project/
├── .dvc/                    # DVC configuration
│   ├── config              # DVC config file
│   └── cache/              # Local cache (gitignored)
├── dvc_storage/            # Local remote storage
├── data/
│   └── raw/
│       ├── insurance_data.csv      # Actual data (gitignored)
│       └── insurance_data.csv.dvc  # DVC metadata (tracked in git)
├── .gitignore              # Updated to exclude data files
└── ...
```

## Troubleshooting

### DVC not found
```bash
pip install dvc
```

### Remote storage not configured
```bash
dvc remote add -d localstorage ./dvc_storage
```

### Data files not found after clone
```bash
dvc pull
```

### Check DVC configuration
```bash
cat .dvc/config
```

## Best Practices

1. **Always commit .dvc files** - These are small metadata files that should be version controlled
2. **Never commit actual data files** - Let DVC handle data versioning
3. **Use descriptive commit messages** - Help track data changes over time
4. **Regular dvc push** - Keep remote storage updated
5. **Document data sources** - Add README files in data directories explaining data sources

## References

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC Best Practices](https://dvc.org/doc/user-guide/best-practices)

