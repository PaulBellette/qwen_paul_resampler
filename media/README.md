# Generated media

`media/` is intended for the small, publishable figures/animations generated from tracked result JSON.

Generate them with:

```bash
uv sync --extra media
uv run python scripts/render_demo.py \
  results/poc_v1/watermark_benchmark.json \
  --out-dir media
```

The renderer does **not** rerun any models. It reads the frozen benchmark JSON only.
