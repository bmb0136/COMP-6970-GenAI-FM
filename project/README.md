# LLM For Recipe Generation

## Setup

Tested python version: 3.13

Libraries:
- numpy
- matplotlib
- pandas
- pytorch
- transformers
- datasets
- accelerate
- trl

## Running

1. Run `sft.py` to train our model (Requires around 6GB VRAM minimum)
2. Run `plot_loss.py` to generate loss plots
3. Run `evaluate.py` to evaluate performance (Requires around 1.5GB VRAM minimum)
4. Use `frontend.py` to try out the trained model
