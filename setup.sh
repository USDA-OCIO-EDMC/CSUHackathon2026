#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="Hackathon2026"

echo "=== Hackathon2026 Environment Setup ==="

# Install Miniconda if conda is not found
if ! command -v conda &>/dev/null; then
    echo "conda not found. Downloading and installing Miniconda..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_INSTALLER="/tmp/miniconda_install.sh"
    curl -fsSL "${MINICONDA_URL}" -o "${MINICONDA_INSTALLER}"
    bash "${MINICONDA_INSTALLER}" -b -p "${HOME}/miniconda3"
    rm "${MINICONDA_INSTALLER}"
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    conda init bash
    echo "Miniconda installed. Reloading shell environment..."
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

# Create or update the conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists, updating..."
else
    echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
    conda create -n "${ENV_NAME}" python=3.11 -y
fi

# Run all installs inside the environment
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "Installing conda-forge packages..."
conda install -c conda-forge \
    geopandas rasterio shapely pyproj earthpy folium \
    numpy pandas matplotlib seaborn tqdm scikit-learn xgboost \
    -y

echo "Installing pip packages..."
pip install \
    torch \
    transformers \
    huggingface_hub \
    requests \
    earthaccess \
    boto3 \
    awscli

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
