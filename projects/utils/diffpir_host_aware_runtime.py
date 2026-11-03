"""A80 independent policy binding on the unchanged official DiffPIR sampling AST."""
import types
from unittest.mock import patch
from utils import diffpir_multiplier_runtime as original
from utils import diffpir_numeric_runtime as numeric
from utils import host_aware_shrink as policy
reference=numeric.reference
CircularBlur=numeric.CircularBlur
schedule=numeric.schedule
stable_solution=numeric.stable_solution


def run(model,y,kernel,case,arm,count,*,steps=100,progress=None):
    if arm=='DIFFPIR_REFERENCE':
        return numeric.run(model,y,kernel,case,'STABLE_DIFFPIR',count,steps=steps,progress=progress)
    if arm not in ('HOST_AWARE','HOST_ONES'):raise ValueError('Unknown A80 arm')
    _,sr,_,_=reference();pending={}
    def solve(clean,*args):
        if pending:raise RuntimeError('Prior quadratic energy not consumed')
        prox=stable_solution(clean,*args)
        rho=args[-2]
        pending['prior_energy']=float(rho)*float((prox-clean).double().square().sum())
        count['module_q_norms']=count.get('module_q_norms',0)+1
        return prox
    def select(dot,r2,d2,n,sigma):
        energy=pending.pop('prior_energy')
        return policy.select(dot,r2,d2,n,sigma,energy)
    mapped='ONES' if arm=='HOST_ONES' else 'SHRINK'
    # Restore both bindings on success or error; shared historical source files stay byte-identical.
    with patch.object(sr,'data_solution',solve),patch.object(original,'policy',types.SimpleNamespace(select=select)):
        out,detail=original.run(model,y,kernel,case,mapped,count,steps=steps,progress=progress)
    if pending:raise RuntimeError('Unconsumed host energy')
    for row in detail['trace']:row['solver_variant']='STABLE_EQUIVALENT_FLOAT32_SF1'
    return out,detail
