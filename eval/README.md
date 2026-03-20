# EigenBench Evaluation Setup

This directory contains everything needed to evaluate your models using the EigenBench framework.

## What's Included

1. **EigenBench Repository** (`external/EigenBench/`)
   - Git submodule pinned to a specific commit
   - Contains the Bradley-Terry-Davison + EigenTrust pipeline
   - Provides vLLM-optimized collection for local models + OpenRouter support

2. **Evaluation Notebook** (`eigenbench_collection_with_external_judges.ipynb`)
   - Uses the official EigenBench mixed-model collection notebook
   - Configured for your local models + external evaluators
   - Generates responses, reflections, and pairwise comparisons
   - Runs Bradley-Terry-Davison fitting and EigenTrust aggregation

3. **Configuration** 
   - `../configs/experiment.yaml` and `experiment.qwen7b.local.yaml` point to `external/EigenBench`
   - `../outputs/eigenbench_runs/kindness_eval/spec.py` is your evaluation spec

4. **Scenario Dataset** (`../data/scenarios/reddit_questions.json`)
   - 300 prompts extracted from your training data
   - Used as evaluation scenarios

### Restoring AskReddit Questions (Kaggle)

If `data/scenarios/reddit_questions.json` was overwritten by synthetic prompts, regenerate it from the Kaggle dataset:

- Dataset: https://www.kaggle.com/datasets/rodmcn/askreddit-questions-and-answers
- Required file: `reddit_questions.csv` (semicolon-delimited)

```bash
python data_gen/build_askreddit_scenarios.py \
   --input-csv /path/to/reddit_questions.csv \
   --output-json data/scenarios/reddit_questions.json \
   --count 300 \
   --min-score 10
```

This writes a JSON array of natural-language AskReddit questions that the evaluation notebook and `spec.py` already expect.

Alternative (auto-download via kagglehub):

```bash
pip install kagglehub
python3 data_gen/download_and_build_askreddit_scenarios.py --count 300 --min-score 10
```

## Two-Model Evaluation Setup

The evaluation compares:

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (baseline, no finetuning)
- **Student (Kind-Finetuned)**: Your finetuned model on kind teacher demonstrations

With external judges:
- **GPT 4.1**: Powerful proprietary evaluator
- **Claude Sonnet 4**: Powerful proprietary evaluator

The base model and finetuned student are evaluated by having GPT-4.1 and Claude Sonnet judge their responses on kindness criteria.

## How to Run

### Prerequisites

```bash
# Initialize the submodule (if not already done)
git submodule update --init --recursive

# Install Python dependencies (use your workspace venv)
pip install torch transformers vllm tqdm python-dotenv scipy scikit-learn matplotlib openai anthropic
```

### Set Up OpenRouter API Key

You only need an **OpenRouter API key**. OpenRouter proxies requests to OpenAI, Anthropic, and other providers.

**Get your key:** https://openrouter.ai/keys

**Option 1: Export in shell**
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Option 2: Create `.env` file in `external/EigenBench/`**
```bash
# external/EigenBench/.env
OPENROUTER_API_KEY=sk-or-v1-...
```

The notebook will use OpenRouter for both:
- GPT-4 Turbo
- Claude 3.5 Sonnet

### Run the Evaluation

```bash
# Open the notebook
jupyter notebook eval/eigenbench_collection_with_external_judges.ipynb
```

**Notebook Execution:**
1. Set up HF cache and install dependencies
2. Clone/navigate to EigenBench
3. Load your spec (2 local models + 2 external judges via OpenRouter)
4. **Phase 1**: Generate responses from Base Model and Student (Kind-Finetuned) on 300 scenarios
5. **Phase 2**: OpenRouter judges (GPT-4 Turbo & Claude 3.5 Sonnet) reflect on each response
6. **Phase 3**: Pairwise comparisons - judges pick which response better aligns with kindness
7. Train Bradley-Terry-Davison + EigenTrust to aggregate judge opinions into alignment scores

### Result Interpretation

The notebook will save results to your spec's `run_root`:
- `evaluations.jsonl`: Raw comparison transcripts
- `eigenbench.png`: Radar chart of alignment scores across kindness criteria
- `eigentrust.txt`: Overall alignment scores for each model
- `uv_embeddings_pca.png`: 2D visualization of model similarity

**Key Output**: Your Student (Kind-Finetuned) model's alignment score vs the Base Model, as judged by powerful external models (GPT-4 Turbo & Claude 3.5 Sonnet) across 8 kindness criteria.

## Customization

To modify the evaluation:

1. **Change scenario count**: Edit `spec.py` → `dataset.count` (default: 300)
2. **Add/remove models**: Edit `spec.py` → `models` dict
   - Local models: `"hf_local:<huggingface-path>"`
   - OpenRouter models: `"openai/gpt-4"`, `"anthropic/claude-sonnet-4"`, etc.
3. **Adjust judges**: Change which models judge by tuning `collection.group_size` and `collections.groups`
4. **Tune training**: Modify `training` settings (dims, lr, max_epochs, etc.)

## Output Structure

Results will be saved to `outputs/eigenbench_runs/kindness_eval/`:

```
kindness_eval/
├── spec.py                    # Your run configuration
├── evaluations.jsonl          # Raw transcript: all responses, reflections, comparisons
└── btd_d2/                    # Training outputs (2-dimensional embedding)
    ├── model.pt               # Trained BTD model weights
    ├── eigenbench.png         # Radar chart: alignment scores per criterion
    ├── uv_embeddings_pca.png  # 2D plot of model positions
    ├── eigentrust.txt         # Alignment scores (Base vs Student)
    ├── training_loss.png      # BTD training curves
    └── log_train.txt          # Full training log
```

### Key Metrics

- **eigenbench.png**: Shows Base Model vs Student across 8 kindness criteria
- **eigentrust.txt**: Consensus alignment score for each model (0.0-1.0)
- **uv_embeddings_pca.png**: How similar the models are in behavioral space

## Quick Reference

**Did collection already run but need to re-fit the model?**
```bash
cd external/EigenBench
python scripts/run.py ../../outputs/eigenbench_runs/kindness_eval/spec.py
```

**Want to check raw evaluation transcripts?**
```bash
# View pairwise comparison decisions
head outputs/eigenbench_runs/kindness_eval/evaluations.jsonl
```

**Need to update EigenBench to latest?**
```bash
cd external/EigenBench
git fetch origin main
git checkout origin/main
cd ../..
git add external/EigenBench
git commit -m "Pin EigenBench to latest"
```

## Important Notes

- **Cost**: Using OpenRouter for GPT-4 Turbo and Claude 3.5 Sonnet as judges will have API costs. OpenRouter typically costs less than calling APIs directly. Budget roughly $3-15 per evaluation depending on scenario count (300 scenarios × ~40 judge calls = ~12k tokens per model).
- **Timeout**: Evaluations can take 2-6 hours depending on OpenRouter queue times and scenario count.
- **Reproducibility**: Seed is fixed in `spec.py` for reproducible scenario/judge sampling.
- **OpenRouter Benefits**: Single API key for multiple providers, often cheaper than direct API access, no separate auth needed.
- **Reproducibility**: Seed is fixed in `spec.py` for reproducible scenario/judge sampling.

## References

- [EigenBench GitHub](https://github.com/jchang153/EigenBench)
- [EigenBench Paper](https://arxiv.org/abs/2509.01938)
- [Mixed OpenRouter + Local Collection Notebook](https://github.com/jchang153/EigenBench/blob/main/notebooks/mixed_openrouter_local_collection.ipynb)
