# Team 34 Windows/VS Code notes

The complete graded Part 3 implementation is `notebooks/p3_agents.ipynb` and its source is `src/p3_agents_source.py`.
The notebook was written for a GPU runtime and includes the full LangGraph pipeline, real safety checker, CLIP critique,
15 scenarios, 10 red-team cases, trace reuse, and rating sheets.

For the official full evaluation, use the lab GPU runtime and run the notebook top to bottom. Keep the evaluation CSVs
unchanged. The public Team 34 Part 2 rank-32 adapter is recorded in `outputs/p2/hf_links_s5000.json`.

The existing `app.py` smoke-test package is not included here; this package is the complete evaluation implementation.
Do not copy the partner's generated results or ratings as your own. Run the notebook and fill the rating sheets with your
own independent judgments.
