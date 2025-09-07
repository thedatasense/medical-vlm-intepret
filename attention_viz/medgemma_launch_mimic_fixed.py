#!/usr/bin/env python3
"""
Compatibility alias for Colab notebooks expecting `medgemma_launch_mimic_fixed`.

This module re-exports `load_model_enhanced` from `medgemma_enhanced.py` so that
existing notebooks can keep their import lines unchanged:

    from medgemma_launch_mimic_fixed import load_model_enhanced

If `medgemma_enhanced` is missing, a clear ImportError is raised with guidance
to add the repo path to `sys.path` in Colab.
"""

try:
    from medgemma_enhanced import load_model_enhanced  # noqa: F401
except Exception as e:  # pragma: no cover - import-time guidance
    raise ImportError(
        "Could not import `medgemma_enhanced`.\n"
        "Please ensure your notebook adds the repo directory to sys.path, e.g.:\n"
        "\n"
        "    import sys\n"
        "    sys.path.insert(0, '/content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz')\n"
        "    %cd /content/drive/MyDrive/Robust_Medical_LLM_Dataset/attention_viz\n"
        "\n"
        "Then retry: from medgemma_launch_mimic_fixed import load_model_enhanced"
    ) from e

