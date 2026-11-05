# Offline skill retrieval evaluation

This check measures the real local `find_skills` index without starting B2E or
calling an LLM. It loads every generated catalog on top of the standard
Heimdall catalog and searches with verified basket queries.

From the repository root:

```bash
cp retrieval_eval/.env.example retrieval_eval/.env
# Set GENERATED_SKILLS_DIR to the collection being evaluated.
./.venv/bin/python -m retrieval_eval
```

Configuration uses the same path variables as the benchmark:

- `HEIMDALL_SKILLS_ROOT`: standard Heimdall skills;
- `CASES`: basket `gold_dataset` root;
- `GENERATED_SKILLS_DIR`: one catalog or a directory containing several catalogs.

Relative paths are resolved from the repository root. CLI arguments `--base`,
`--cases`, and `--generated` override the local configuration. Output goes to a
timestamped directory under `retrieval_eval/results/`.

The report contains positive-score Recall@1/3/5/10 and MRR for both the exact
generated skill and its logical family, plus a row-level CSV with the top five
results for each query.
Use `--category no_data` or `--category access_control` to select other case
categories; repeat the flag to combine categories.
