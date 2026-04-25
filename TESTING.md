# Testing Guide for Phase 3 Pipeline

## Quick Start

### **Option 1: Run All Tests (Recommended for Quick Check)**

```bash
cd /Users/cailianzhu/Documents/Hak/Hackathon2026

# Mock tests (NO API calls needed - runs in ~2-5 minutes)
python test_phase3.py --mode mock
```

### **Option 2: Run Unit Tests**

```bash
# Tests individual components
python test_phase3.py --mode unit
```

### **Option 3: Run Real API Tests**

```bash
# Requires NASA credentials
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password

python test_phase3.py --mode real
```

---

## What Each Test Mode Does

### **MOCK MODE** ✅ Recommended (No APIs)
- ✓ Tests Prithvi model loading
- ✓ Creates synthetic HLS data
- ✓ Tests feature extraction on synthetic data
- ✓ Tests weather vector synthesis
- ✓ Tests feature fusion (weather + Prithvi)
- ✓ Tests analog year identification with mock data
- ✓ Runs full pipeline end-to-end with synthetic data

**Time:** ~2-5 minutes  
**Prerequisites:** None (just PyTorch, pandas, numpy)  
**Best for:** Quick validation before running real APIs

---

### **UNIT MODE** 
- Tests individual function loading and basic functionality
- Attempts API calls (weather, NASS)
- May skip if APIs unavailable

**Time:** ~1-2 minutes  
**Prerequisites:** Optional NASA credentials

---

### **REAL MODE**
- Searches actual HLS granules from NASA CMR
- Fetches real weather from NASA POWER
- Downloads satellite data

**Time:** ~5-10 minutes (depends on API response)  
**Prerequisites:** 
  - EARTHDATA_USERNAME
  - EARTHDATA_PASSWORD
  - Register at: https://urs.earthdata.nasa.gov/

---

## Expected Output

### Mock Test Success (What You Should See):

```
======================================================================
PHASE 3 TEST SUITE - Mode: MOCK
======================================================================

>>> RUNNING MOCK TESTS (No API calls) <<<

[MOCK TEST 1] Prithvi Extraction with Synthetic Data
------------------------------------------------------------
  Creating synthetic HLS tile (6, 224, 224)...
  Loading Prithvi model...
  Extracting features...
  ✓ Embedding shape: (768,)
  ✓ Embedding type: <class 'numpy.ndarray'>
  ✓ Sample values: [-0.35 -0.23  0.12 ...
  ✓ Dimension check passed

[MOCK TEST 2] Weather Vector Synthesis
------------------------------------------------------------
  Creating synthetic weather vectors...
  ✓ Created 10 weather vectors
  ✓ Shape: (10, 18)
  ✓ Sample vector: [-0.52  0.81  0.34 ...]
  ✓ Dimension check passed

[MOCK TEST 3] Feature Fusion (Weather + Prithvi)
------------------------------------------------------------
  Creating synthetic weather (18-dim) + Prithvi (768-dim)...
  Weather shape: (18,)
  Prithvi shape: (768,)
  ✓ Combined shape: (786,)
  ✓ Expected: (786,)
  ✓ Dimension check passed

[MOCK TEST 4] Analog Year Identification
------------------------------------------------------------
  Creating mock feature space...
  ✓ Created feature space: 126 records
  ✓ Created yield data: 20 records
  Running analog year identification...
  ✓ Got results: 5 analog years

  Results:
     year  similarity  yield_bu_acre
     2012      0.9789          167.3
     2008      0.9234          172.1
     2015      0.9102          170.5
     2019      0.8987          168.9
     2011      0.8765          175.2

  ✓ Top analog year: 2012 (similarity: 0.979)

[INTEGRATION TEST] Full Mock Pipeline
============================================================
Step 1: Create mock Prithvi embeddings for all counties...
  ✓ Created 72 Prithvi records
Step 2: Create mock weather vectors...
  ✓ Created weather for 2 counties
Step 3: Fuse features (Weather + Prithvi)...
  ✓ Created 72 fused records
  ✓ Embedding dimension: (786,)
Step 4: Run analog year identification...
  ✓ Identified 5 analog years

============================================================
✓ FULL MOCK PIPELINE TEST PASSED
============================================================

======================================================================
TEST SUMMARY
======================================================================
  ✓ PASS   mock_prithvi
  ✓ PASS   mock_weather
  ✓ PASS   mock_fusion
  ✓ PASS   mock_analog
  ✓ PASS   full_pipeline

Total: 5 passed, 0 skipped, 0 failed
======================================================================

✓ All tests passed!
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'privthi_extractor'`

**Fix:** Run from project root directory:
```bash
cd /Users/cailianzhu/Documents/Hak/Hackathon2026
python test_phase3.py --mode mock
```

---

### Issue: `RuntimeError: CUDA out of memory`

**Fix:** Use CPU instead:
```bash
# Tests already default to CPU, but if needed:
export CUDA_VISIBLE_DEVICES=""
python test_phase3.py --mode mock
```

---

### Issue: `ImportError: No module named 'torch'`

**Fix:** Install dependencies:
```bash
conda activate Hackathon2026
pip install torch transformers huggingface_hub
```

---

### Issue: `EARTHDATA_USERNAME not set` (in real tests)

**Fix:** Set environment variables:
```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password

# Or add to ~/.bashrc / ~/.zshrc for persistence
echo 'export EARTHDATA_USERNAME=your_username' >> ~/.zshrc
echo 'export EARTHDATA_PASSWORD=your_password' >> ~/.zshrc
source ~/.zshrc
```

Register for free: https://urs.earthdata.nasa.gov/

---

## Testing Workflow

### **Phase 1: Quick Validation (5 minutes)**
```bash
# Run mock tests to ensure code structure is correct
python test_phase3.py --mode mock
```

### **Phase 2: Component Testing (if mock passes)**
```bash
# Test individual functions
python test_phase3.py --mode unit
```

### **Phase 3: End-to-End with Real Data (if unit passes)**
```bash
# Run full pipeline with actual NASA APIs
export EARTHDATA_USERNAME=<your_username>
export EARTHDATA_PASSWORD=<your_password>

python test_phase3.py --mode real
```

---

## Interpreting Test Results

| Status | Meaning | Action |
|--------|---------|--------|
| ✓ PASS | Test succeeded | Continue to next test |
| ⊘ SKIP | Test skipped (API unavailable, etc.) | Not a failure; API issue |
| ✗ FAIL | Test failed | Debug and fix |

---

## Next: Debug Individual Components

If tests fail, debug specific modules:

### Debug Prithvi Loading
```python
python -c "
import sys
sys.path.insert(0, 'src')
from privthi_extractor import load_prithvi
model = load_prithvi(device='cpu')
print('✓ Prithvi loaded')
"
```

### Debug Weather API
```python
python -c "
import sys
sys.path.insert(0, 'src')
from analog_years import get_county_weather_features_single
weather = get_county_weather_features_single(lat=42.0, lon=-93.6, year=2024, forecast_date='final')
print('✓ Weather fetched, shape:', weather.shape)
"
```

### Debug Feature Fusion
```python
python -c "
import sys
import numpy as np
sys.path.insert(0, 'src')

# Create mock data
weather = np.random.randn(18).astype(np.float32)
prithvi = np.random.randn(768).astype(np.float32)

# Fuse
combined = np.concatenate([weather, prithvi])
print('✓ Fused embedding shape:', combined.shape)
"
```

---

## Running Tests in Jupyter Notebook

Instead of terminal, you can run tests in a notebook:

```python
import subprocess
result = subprocess.run(
    ["python", "test_phase3.py", "--mode", "mock"],
    cwd="/Users/cailianzhu/Documents/Hak/Hackathon2026",
    capture_output=True,
    text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
```

---

## Summary

| Test Mode | Speed | Needs APIs | Best For |
|-----------|-------|-----------|----------|
| **Mock** | ⚡ 2-5 min | No | Quick validation |
| **Unit** | ⚡ 1-2 min | Optional | Component checks |
| **Real** | 🐢 5-10 min | Yes | Production validation |

**Recommended: Start with `mock`, then `real` when ready.**
