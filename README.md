# Language Models use Lookbacks to Track Beliefs

This repository contains the code and experiments for the paper ["Language Models use Lookbacks to Track Beliefs"](https://arxiv.org/abs/2505.14685) by Prakash et al. The work investigates how language models (specifically Llama-3-70B-Instruct) represent and track characters' beliefs using.

## Repository Structure

```
.
├── data/               # Dataset files
├── experiments/        # Experiment configurations and results
├── notebooks/         # Jupyter notebooks for analysis
├── scripts/           # Utility scripts
├── src/              # Source code
│   ├── dataset.py    # Dataset implementation
│   └── env_utils.py  # Environment utilities
├── svd/              # SVD analysis code
├── svd_results/      # Results from SVD analysis
└── env.yml           # Environment configuration
```

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Set up the environment:
```bash
conda env create -f env.yml
conda activate [env-name]
```

3. Configure environment variables:
- Set `NDIF_KEY` for API access
- Set `HF_WRITE` for Hugging Face access

## Usage

The repository contains several components:

1. **Dataset Generation**: The `src/dataset.py` file contains code for generating and processing the belief tracking dataset.

2. **Experiments**: The `experiments/` directory contains configurations and results for various experiments investigating the lookback mechanism.

3. **Analysis**: 
   - SVD analysis code is in the `svd/` directory
   - Results are stored in `svd_results/`
   - Jupyter notebooks in `notebooks/` provide interactive analysis

4. **Utilities**: Various utility scripts are available in the `scripts/` directory.

## Key Findings

The paper demonstrates that language models use a "lookback mechanism" to track characters' beliefs. This mechanism:
- Binds character-object-state triples using Ordering IDs (OIs)
- Uses visibility IDs to encode relationships between characters
- Implements lookbacks to retrieve and update belief information

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{prakash2024language,
  title={Language Models use Lookbacks to Track Beliefs},
  author={Prakash, Nikhil and Shapira, Natalie and Sharma, Arnab Sen and Riedl, Christoph and Belinkov, Yonatan and Shaham, Tamar Rott and Bau, David and Geiger, Atticus},
  journal={arXiv preprint arXiv:2505.14685},
  year={2024}
}
```

## License

[Add appropriate license information]

## Contact

For questions and issues, please open an issue in this repository or contact the authors. 