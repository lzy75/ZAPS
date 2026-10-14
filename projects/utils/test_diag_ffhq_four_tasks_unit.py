"""Four-task protocol contracts; optional actual operator/VJP tests, no UNet."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from utils import diag_ffhq_four_tasks_unit as audit


class ContractTests(unittest.TestCase):
    def test_paper_task_numbers_configs_and_no_tuned_lr(self):
        self.assertEqual(audit.TASKS,("gaussian_deblur","inpainting","motion_deblur","super_resolution"))
        self.assertEqual([audit.PAPER[t]["psnr"] for t in audit.TASKS],[26.06,27.79,24.16,26.63])
        for task in audit.TASKS:
            c=audit.config_for(task)
            self.assertEqual(c["lr"],.001);self.assertEqual(c["d_init"],.2)
            self.assertEqual(c["zeta_init"],.2 if task.endswith("deblur") else .1)
            self.assertEqual(c["schedule"],[15,10,5])
            self.assertEqual((c["num_steps"],c["num_epochs"]),(30,10))
            self.assertFalse(c["surrogate_score_jacobian"] or c["use_learned_var"])
        with self.assertRaises(ValueError):audit.config_for("box_inpainting")

    def test_coverage_required_and_paper_gaps_are_contextual(self):
        rows=[dict(task=t,psnr=audit.PAPER[t]["psnr"]+.2,ssim=audit.PAPER[t]["ssim"],
            lpips=audit.PAPER[t]["lpips"],float_psnr=25.) for t in audit.TASKS]
        result=audit.summarize(rows,1)
        for r in result:
            self.assertEqual((r["images"],r["paper_images"]),(1,1000))
            self.assertIsNone(r["psnr_std"])
            self.assertAlmostEqual(r["delta_psnr_vs_paper_mean"],.2)
        with self.assertRaises(ValueError):audit.summarize(rows[:-1],1)
        with self.assertRaises(ValueError):audit.summarize(rows,10)

    def test_explicit_subset_budget_no_hidden_expansion(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            for i in range(3):(folder/f"{i:05d}.png").touch()
            selected=audit.select_images(None,name,2,1)
            self.assertEqual([p.name for p in selected],["00001.png","00002.png"])
            self.assertEqual(len(audit.select_images(str(folder/"00000.png"),None,None,0)),1)
            for count,start in ((None,0),(21,0),(4,0),(2,-1)):
                with self.assertRaises(ValueError):audit.select_images(None,name,count,start)
            with self.assertRaises(ValueError):audit.select_images(str(folder/"00000.png"),None,10,0)

    def test_no_state_extra_sample_or_gt_selection(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        names={getattr(n.func,"attr",getattr(n.func,"id",None)) for n in calls}
        self.assertFalse(names & {"sample","p_sample_loop","scheduler_for","TransposeMode"})
        optimize=[n for n in calls if getattr(n.func,"attr",None)=="optimize"]
        self.assertEqual(len(optimize),1)
        self.assertTrue(any(k.arg=="x0_gt" and isinstance(k.value,ast.Constant) and k.value.value is None
                            for k in optimize[0].keywords))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy","PIL","scipy")),
    "NumPy/Pillow/SciPy unavailable; original motion generation test NOT run")
class MotionTests(unittest.TestCase):
    def test_original_motion_reproducible_finite_normalized_and_rng_preserved(self):
        import numpy as np
        np.random.seed(1234);state=np.random.get_state()
        a=audit.original_motion_kernel(0);after=np.random.get_state()
        self.assertEqual(state[0],after[0]);np.testing.assert_array_equal(state[1],after[1]);self.assertEqual(state[2:],after[2:])
        b=audit.original_motion_kernel(0)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(61,61));self.assertTrue(np.isfinite(a).all())
        self.assertGreaterEqual(float(a.min()),0.);self.assertAlmostEqual(float(a.sum()),1.,places=6)


@unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch unavailable; exact-adjoint/mask/gradient tests NOT run")
class TensorTests(unittest.TestCase):
    def test_all_four_affine_and_exact_vjp_gates_mask_frozen_without_rng(self):
        import torch
        import numpy as np
        gt=torch.linspace(-.8,.8,3*64*64).reshape(1,3,64,64)
        direction=torch.cos(torch.arange(gt.numel()).float()).reshape_as(gt)
        kernel=np.zeros((61,61),dtype=np.float32);kernel[30,30]=.7;kernel[28,32]=.3
        for task in audit.TASKS:
            before=torch.get_rng_state().clone()
            base=audit.make_base(task,"cpu",gt,2001000,kernel)
            operator=audit.unit_operator(base)
            a=operator.H(gt);b=operator.H(gt)
            self.assertTrue(torch.equal(a,b))
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            gate=audit.operator_gate(operator,gt,direction)
            self.assertTrue(gate["passed"],(task,gate))
            if task=="inpainting":
                self.assertEqual(base._mask_cache.shape,(1,1,64,64))
                self.assertGreater(float(1-base._mask_cache.mean()),.65)
                self.assertLess(float(1-base._mask_cache.mean()),.75)

    def test_motion_asymmetric_adjoint_not_forward(self):
        import torch
        import numpy as np
        kernel=np.zeros((61,61),dtype=np.float32);kernel[30,30]=.7;kernel[28,32]=.3
        base=audit.motion_operator(kernel,"cpu")
        r=torch.cos(torch.arange(64*64).float()).reshape(1,1,64,64)
        self.assertFalse(torch.allclose(base.H(r),base.transpose(r)))
        unit=audit.unit_operator(base)
        torch.testing.assert_close(unit.transpose(r),.5*base.transpose(r))


if __name__=="__main__":unittest.main()
