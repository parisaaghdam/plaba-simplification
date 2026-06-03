# PLABA Simplification System (LangGraph + Pydantic)

Minimal multi-agent medical text simplification with iterative refinement:
- `analyze` agent identifies complex words, extracts medical concepts, flags plain-language issues in the source, and builds a glossary plus `medical_terms`.
- `simplifier` agent rewrites biomedical text using PLAIN-style criteria (common words, short sentences, active voice, defined acronyms, etc.), the glossary, and preservation list.
- `quality_gate` agent merges faithfulness, plain-language checklist, readability metrics (FKGL, Flesch ease), automatic plain-language heuristics (sentence length, acronyms, passive voice), and optional SARI to accept or request revision.
- LangGraph loops `simplifier` → `quality_gate` until accepted or `max_iterations` reached.

The `simplifier` can use a model fine-tuned on PLABA (e.g. Mistral-7B via Ollama). See [`training/README.md`](training/README.md) for the data prep, QLoRA training, and integration steps.

## Dataset Downloaded

Files are in `data/plaba`:
- `data.json`
- `train.csv`
- `val.csv`
- `test.csv`
- `README.pdf`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your API key:

```bash
set OPENAI_API_KEY=your_key_here
```

## Run

Simplify one text:

```bash
python main.py --text "The patient exhibited acute pharyngitis..."
```

Evaluate on 20 random sentence-level samples from PLABA validation split:

```bash
python main.py --eval --data data/plaba/val.csv
```

Save outputs to a custom experiment directory:

```bash
python main.py --eval --data data/plaba/val.csv --output-dir outputs/experiments
```

## Notes

- Evaluation currently reports:
  - average SARI (when references are available in the eval split)
  - average Flesch-Kincaid grade
  - quality-gate acceptance rate
- Each evaluation run now writes:
  - `eval-<timestamp>.json` (metrics + per-sample rows)
  - `eval-<timestamp>.csv` (flat per-sample table)
- PLABA has multiple references for many source texts; evaluation groups these as gold references.
