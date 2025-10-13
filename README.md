# Evolution VLLM

Implementation of the algorithms in [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/pdf/2509.24372) paper with SGLang as the inference engine for improved throughput on a single GPU. 

The default code from the paper uses Huggingface's Transformers Library, which has a slow and unoptimized generate function. SGLang's generation speed is around 4 times faster for large batch sizes.

# Quickstart: 
The following commands trains Qwen2.5-7B-Instruct to perform 4 digit multiplication. 
```
pip install -r requirements.txt
python generate_dataset.py
python evolve.py
```

# Features
This repository is significantly better for single GPU usage compared to the repository linked in the paper. Some reasons why you might want to use this repository are below. 
- Using SGLang improves inference speed by roughly 4 times. 
- The inference engine remains initialized across model evaluations, eliminating repeated startup overhead.
- Evolutionary algorithms don't require gradients, so you can full-rank fine-tune a 7B model on a card with 24GB of VRAM (RTX 3090 or RTX 4090). 
- Evolutionary algorithms are less prone to reward hacking and often perform better than RL, as observed in [this paper](https://arxiv.org/pdf/2509.24372). 
- Evolutionary algorithms are less sensitive to hyperparameters. 
- Evolutionary algorithms don't use a KL divergence term, so there's no need to store a copy of the base model. 
- We provide a straightforward configuration file and easily customizable reward function that you can use with any task. 

# Code Structure
All data is stored as a list of samples, where each sample is a dictionary with a question and answer. The reward function is implemented in reward.py and easily customizable. Training can be easily modified by the configuration file in conf/config.yaml. 

# Other Notes
- If there are any bugs, please open an issue. 
- Tested with Python 3.10