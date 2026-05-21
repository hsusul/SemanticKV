# Documentation Assets

This directory is for small, representative images used in GitHub-facing documentation.

Generated experiment plots are written under:

```text
outputs/experiments/<timestamp>_<experiment_id>/
```

Useful plots to copy here for README or report screenshots:

- `policy_comparison_avg_ttft.png`
- `adaptive_weights_over_time.png`
- `evictions_by_semantic_type.png`
- `memory_utilization_over_time.png`

Example:

```bash
cp outputs/experiments/<timestamp>_low_memory_rag_stress/policy_comparison_avg_ttft.png docs/assets/low_memory_rag_avg_ttft.png
cp outputs/experiments/<timestamp>_workload_shift_static_vs_adaptive/adaptive_weights_over_time.png docs/assets/workload_shift_adaptive_weights.png
cp outputs/experiments/<timestamp>_low_memory_rag_stress/evictions_by_semantic_type.png docs/assets/low_memory_rag_evictions.png
```

The benchmark outputs directory is ignored by git, while selected assets in this directory can be committed when they are useful for GitHub display.
