"""Reuse A102 mechanics with the case's frozen observation noise."""
from contextlib import contextmanager
from unittest.mock import patch
from utils import shared_prefix_feedback_runtime as base
make_operator=base.make_operator
train_prefix=base.train_prefix
@contextmanager
def case_sigma(value):
 with patch.object(base.rules,'SIGMA',float(value)):yield
def run_suffix(model,operator,y,case,arm,prefix):
 with case_sigma(case['sigma']):return base.run_suffix(model,operator,y,case,arm,prefix)
