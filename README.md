# autogluon_waveform

## Overview

This repository implements the **Auto-Unrolled Proximal Gradient Descent (PGD) network** for MISO beamforming, with integration of AutoGluon for hyperparameter optimization (HPO) and comparison against classical and black-box baselines. The project is designed for research in wireless communications, specifically for optimizing beamforming in 6G MISO systems.

## Key Components

- **AutoGluon Integration** (`autogluon_code.py`):  
	- Bayesian HPO outer-loop: Searches over network depth, step-size initialization, optimizer, and learning rate scheduler for the best Auto-PGD configuration.
	- TabularPredictor baseline: Trains a supervised AutoML model on Zero-Forcing (ZF) labels for comparison.

- **Baselines** (`baselines.py`):  
	- Zero-Forcing (ZF): Closed-form solution, no training.
	- Classical PGD solver: Reference implementation with fixed iterations.
	- Black-box MLP: 5-layer PyTorch MLP trained with sum-rate loss.
	- Convenience evaluation wrappers.

- **Data Generation** (`data_generator.py`):  
	- Utilities for generating synthetic 6G MISO channel datasets.
	- Implements Frobenius ball projection and sum-rate computation.

- **PGD-Net Baseline** (`PGDNet.py`):  
	- Unrolled PGD network with fixed depth and per-layer learned step sizes.
	- Serves as an ablation baseline to highlight the benefits of AutoGluon HPO.

- **Auto-Unrolled PGD Network** (`unrolled.py`):  
	- Each layer corresponds to a PGD iteration with learnable step-size.
	- Step-sizes and network depth are optimized via AutoGluon.

- **Training & Evaluation** (`train_evaluate.py`):  
	- Unified training loop, experiment orchestration, and results serialization.
	- Supports quick smoke tests and full experiments (CPU/GPU).

## Usage

### Quick Smoke Test

```bash
python train_evaluate.py --smoke-test
```

### Full Experiment

```bash
python train_evaluate.py --full
```

### Full Experiment on GPU

```bash
python train_evaluate.py --full --device cuda
```

## File Structure

- `autogluon_code.py`: AutoGluon integration and baseline comparison.
- `baselines.py`: Classical and black-box baselines.
- `data_generator.py`: Data generation and utility functions.
- `PGDNet.py`: PGD-Net ablation baseline.
- `unrolled.py`: Auto-Unrolled PGD network implementation.
- `train_evaluate.py`: Training, evaluation, and experiment orchestration.
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- PyTorch
- AutoGluon
- NumPy
- Pandas

Install dependencies with:

```bash
pip install torch autogluon numpy pandas
```

## Citation

If you use this codebase in your research, please cite the corresponding paper.

## License

MIT License

Copyright (c) 2026 Ahmet Kaplan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
