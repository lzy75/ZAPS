"""Mandatory Torch gates: production measurement arithmetic and RNG pairing unchanged."""
import unittest
from unittest.mock import patch
import torch
from utils.test_counted_adjoint_repair_torch import NumericTests as CountedOperatorTests
from utils import historical_ffhq_runtime as engine
from utils import counted_adjoint_repair as counter
from utils import paper_clock_runtime as common
from utils.current_score_runtime_r2 import input_vjp_parameters
from utils.test_paper_clock_runtime import TinyDiffusion


class RecordingArithmeticTests(unittest.TestCase):
    def test_same_engine_and_counted_operator_are_reused_for_all_continued_arms(self):
        for arm in ('HISTORICAL_NATIVE','LEGACY_IEEE','EXACT_IEEE','RESPONSE_30'):
            model=TinyDiffusion()
            with counter.installed(),input_vjp_parameters(model.model),common.counters(model.model) as counts:
                op=engine.make_operator('cpu',counts,size=16,mode='exact' if arm=='EXACT_IEEE' else 'legacy_bicubic')
                out,details=engine.execute(model,op,torch.full((1,3,4,4),.15),1003,arm,counts,size=16,level=1)
            engine.validate_counts(counts,arm);self.assertTrue(torch.isfinite(out).all())
            self.assertIsNone(model.model.weight.grad)


if __name__=='__main__':unittest.main()
