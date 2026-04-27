# Optimization Assets

This directory is generated from `PROJECT_OPTIMIZATION_SUMMARY.md`.

## Files

- `optimization_steps.csv`: structured optimization-step data
- `optimization_steps.md`: academic-style Markdown table
- `optimization_steps.tex`: LaTeX table for paper/thesis
- `spinn_vs_ort.tex`: LaTeX comparison table
- `optimization_trajectory.svg`: stage-by-stage latency trajectory
- `optimization_waterfall.svg`: waterfall latency reduction chart
- `spinn_vs_ort.svg`: SPINN vs ONNX Runtime comparison

## Usage

Run:

```bash
python3 scripts/visualize_optimization.py
```

Then insert the generated `.svg` figures into slides and the `.tex` tables into your thesis.
