# FBD Evaluation Approaches: Server vs Test Simulation

## Overview

This document explains the key differences between the two evaluation approaches in the FBD (Federated Block-wise Distillation) codebase and provides guidance on when to use each approach.

## The Problem

Users reported that `fbd_test_sim.py` outputs very low accuracy (~16%) while `fbd_server_sim.py` records high accuracy (~75%) when using the same saved warehouse weights.

## Root Cause Analysis

### Critical Issue: Target Handling

**❌ Original Problem in `fbd_test_sim.py`:**
```python
# WRONG: Ignores targets completely
for inputs, _ in data_loader:
    outputs = model(inputs.to(device))
    # ... process outputs ...

# WRONG: Passes None as targets to evaluator
auc, acc = evaluator.evaluate(scores, None, None)
```

**✅ Fixed in `fbd_test_sim.py`:**
```python
# CORRECT: Properly handles targets
for inputs, targets in data_loader:
    outputs = model(inputs.to(device))
    # ... process both outputs and targets ...

# CORRECT: Uses actual targets for evaluation
auc, acc = evaluator.evaluate(scores, true_targets, None)
```

## Key Differences Between Approaches

### 1. Evaluation Method

| Aspect | Server Simulation | Test Simulation (Fixed) |
|--------|-------------------|-------------------------|
| **Function** | `_test_model()` | `_test_model_with_targets()` |
| **Target Handling** | ✅ Proper target processing | ✅ Proper target processing (after fix) |
| **Loss Calculation** | ✅ Calculates actual loss | ❌ No loss (scores only) |
| **Evaluation Call** | `evaluate(scores, targets, None)` | `evaluate(scores, targets, None)` |

### 2. Warehouse State

| Aspect | Server Simulation | Test Simulation |
|--------|-------------------|-----------------|
| **Warehouse** | Live during training | Loaded from saved file |
| **Model Template** | `use_pretrained=True` | `use_pretrained=False` |
| **Weight Source** | Current training state | Saved checkpoint |

### 3. Evaluation Scope

| Aspect | Server Simulation | Test Simulation |
|--------|-------------------|-----------------|
| **Models Evaluated** | Individual evaluation per call | Comprehensive ensemble analysis |
| **Ensemble Method** | Block-wise hybrid models | Simple majority voting |
| **Output Detail** | Basic metrics | Detailed per-sample analysis |

## When to Use Each Approach

### Use Server Simulation (`fbd_server_sim.py`) When:
- ✅ You want real-time evaluation during training
- ✅ You need block-wise ensemble evaluation
- ✅ You want to monitor training progress
- ✅ You need loss calculations

### Use Test Simulation (`fbd_test_sim.py`) When:
- ✅ You want final evaluation after training
- ✅ You need detailed per-sample analysis
- ✅ You want confidence metrics (c_i, z_i)
- ✅ You need majority voting ensemble
- ✅ You want to save individual predictions

## Fixed Issues in `fbd_test_sim.py`

### 1. ✅ Proper Target Handling
- Added `_test_model_with_targets()` function
- Fixed individual model evaluation to use actual targets
- Fixed ensemble evaluation to use proper target data

### 2. ✅ Diagnostic Tools
- Added `diagnose_warehouse_weights()` function
- Provides detailed warehouse state information
- Helps debug weight loading issues

### 3. ✅ Better Logging
- Added fix notification in logs
- Detailed warehouse diagnostics
- Individual model accuracy reporting

## Expected Results After Fix

After applying the fix, both evaluation approaches should report similar individual model accuracies:

**Before Fix:**
- Server Sim: M0: 74.68%, M1: 71.37%, etc.
- Test Sim: Ensemble: 16.12% (❌ Wrong!)

**After Fix:**
- Server Sim: M0: 74.68%, M1: 71.37%, etc.  
- Test Sim: M0: ~74.68%, M1: ~71.37%, etc. (✅ Correct!)

## Usage Examples

### Server Simulation
```bash
# Real-time evaluation during training
python fbd_server_sim.py --experiment_name organsmnist --model_flag resnet18 ...
```

### Test Simulation (Fixed)
```bash
# Final comprehensive evaluation
python fbd_test_sim.py --experiment_name organsmnist --model_flag resnet18 \
  --final_ensemble --comm_dir fbd_comm --output_dir final_results
```

## Technical Details

### Evaluation Function Comparison

**Server Simulation `_test_model()`:**
```python
for inputs, targets in data_loader:
    outputs = model(inputs.to(device))
    if task == 'multi-label, binary-class':
        targets = targets.to(torch.float32).to(device)
        loss = criterion(outputs, targets)
        m = nn.Sigmoid()
    else:
        targets = torch.squeeze(targets, 1).long().to(device)
        loss = criterion(outputs, targets)
        m = nn.Softmax(dim=1)
```

**Test Simulation `_test_model_with_targets()` (Fixed):**
```python
for inputs, targets in data_loader:
    outputs = model(inputs.to(device))
    if task == 'multi-label, binary-class':
        targets = targets.to(torch.float32).to(device)
        m = nn.Sigmoid()
    else:
        targets = torch.squeeze(targets, 1).long().to(device)
        m = nn.Softmax(dim=1)
        targets = targets.float().resize_(len(targets), 1)
```

## Troubleshooting

### If Test Simulation Still Shows Low Accuracy:

1. **Check Warehouse File:**
   - Ensure the warehouse file exists and is not corrupted
   - Check the diagnostic output for weight loading issues

2. **Verify Model Configuration:**
   - Ensure `--model_flag`, `--in_channels`, `--num_classes` match training config
   - Check that `--experiment_name` matches the saved warehouse

3. **Check Target Shape:**
   - Verify that target dimensions match expected format
   - Check task type (multi-class vs multi-label vs binary)

4. **Compare with Server Results:**
   - Run server simulation on the same data to verify expected accuracy
   - Compare individual model weight checksums

## Conclusion

The key issue was improper target handling in the original `fbd_test_sim.py`. After fixing this issue, both evaluation approaches should provide consistent and accurate results, with each serving different use cases in the FBD workflow. 