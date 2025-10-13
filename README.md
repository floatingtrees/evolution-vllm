# Evolution VLLM

Implementation of the algorithms in [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/pdf/2509.24372) paper with SGLang as the inference engine for improved throughput on a single GPU. 

The default code from the paper uses Huggingface's Transformers Library, which has a slow and unoptimized generate function. SGLang's generation speed is around 4 times faster for large batch sizes.

# Quickstart: 

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python evolve.py
```

# Features
This repository is significantly better for single GPU usage compared to the repository linked in the paper. Some reasons why you might want to use this repository are below. 
- Using SGLang improves inference speed by roughly 4 times. 
- The inference engine stays initialized when evaluating different models, so we don't have wait for startup time. 
- Evolutionary algorithms don't require gradients, so you can full-rank fine-tune a 7B model on a card with 24GB of VRAM (RTX 3090 or RTX 4090)
- Evolutionary algorithms are less prone to reward hacking and converge better, as observed in [this paper](https://arxiv.org/pdf/2509.24372)
