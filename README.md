# Evolution VLLM

Implementation of the [Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning](https://arxiv.org/pdf/2509.24372) paper with SGLang as the inference engine for improved throughput on a single GPU. 

The default code from the paper uses Huggingface's Transformers Library, which has a slow and unoptimized generate function. SGLang's generation speed is around 4x faster for large batch sizes

# Quickstart: 

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python evolve.py
```

