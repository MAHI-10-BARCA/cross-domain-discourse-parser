# Report figures (Chapter 6)

Thesis charts generated from **verified** GUM evaluation metrics in the project logs.

## Generate locally

From the repository root:

```bat
pip install matplotlib numpy
python scripts\generate_report_figures.py
```

This creates:

| File | Description |
|------|-------------|
| `fig6_1_accuracy.png` | Heuristic vs trained relation accuracy (31.5% vs 43.3%) |
| `fig6_2_prf.png` | Macro precision / recall / F1 |
| `fig6_3_f1.png` | Macro-F1 vs weighted-F1 |
| `fig6_4_attachment.png` | Unlabeled (~50%) and labeled (~14%) tree attachment |
| `fig6_5_downstream.png` | Verified metrics + author-reported ~64% downstream tree quality |

## Notes

- Pair-level trained accuracy **43.3%** and attachment F1 values come from `outputs/evaluation/` JSON runs.
- The **~64%** bar is the author-reported downstream tree quality via summarization comparison to gold; keep it distinct from pair-level accuracy in the report and viva.
- After generating, commit the PNGs if you want them tracked:

```bat
git add report_figures\*.png
git commit -m "Add Chapter 6 report figures"
git push
```
