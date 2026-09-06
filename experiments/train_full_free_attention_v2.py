"""Run frozen nanoGPT training with the isolated Attention-Free v2 graph.

Usage:
    python experiments/train_full_free_attention_v2.py \
        config/train_shakespeare_char_full_free_attention_v2_smoke.py
"""

import os
import runpy
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import model as frozen_model
from experiments.full_free_attention_v2 import install_into_frozen_model_module


install_into_frozen_model_module(frozen_model)
runpy.run_path(os.path.join(ROOT, "train.py"), run_name="__main__")

