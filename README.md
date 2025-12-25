# LongMemEval

This is a fork of LongMemEval made compatible with openbench and inspect-ai.

## Usage
Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Create a .env file with
```
HF_TOKEN=hugging_face_token_here

# Pick your provider and enter API key
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```

`uv pip install -e .`

You should then be able to see `longmemeval` listed under community benchmarks when you run:
`uv run bench list`

To run the benchmarks:
`uv run --env-file .env bench eval longmemeval-small --model openai/gpt-5-nano`

## Citation

If you find the work useful, please cite:

```
@article{wu2024longmemeval,
      title={LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory}, 
      author={Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
      year={2024},
      eprint={2410.10813},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10813}, 
}
```
