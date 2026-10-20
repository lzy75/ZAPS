"""A23 frozen scalar-only policy: 30 calls, causal states, bounded noise-coordinate slots."""
import math
import statistics
import numpy as np

ARMS=('C','R','D','RC')
RHO=.25
EPS=1e-12


def noise_u(alpha):
    a=np.asarray(alpha,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all() or not ((a>0)&(a<1)).all() or not np.all(np.diff(a)<0):raise ValueError('Need strictly decreasing cumulative alphas')
    logs=.5*(np.log1p(-a)-np.log(a))
    return (logs[-1]-logs)/(logs[-1]-logs[0])


def reference(records,exclude_image=None):
    keep=[r for r in records if r['case']['image_id']!=exclude_image]
    images=sorted({r['case']['image_id'] for r in keep})
    if len(images)<2:raise ValueError('Need at least two distinct reference images')
    if any(len(r['rows'])!=30 for r in keep):raise ValueError('Reference must use original 30-point rows')
    u=[r['noise_u'] for r in keep[0]['rows']]
    if any([r['noise_u'] for r in rr['rows']]!=u for rr in keep) or any(b<=a for a,b in zip(u,u[1:])):raise ValueError('Reference noise grids differ')
    def curve(key,start):
        values=[]
        for k in range(start,30):
            per_image=[]
            for image in images:
                rows=[r for r in keep if r['case']['image_id']==image]
                if len(rows)!=2 or {r['case']['sampling_seed'] for r in rows}!={1000,2000}:raise ValueError('Need both fixed seeds per reference image')
                v=[r['rows'][k][key] for r in rows]
                if any(x is None or not math.isfinite(x) or x<0 for x in v):raise ValueError('Unresolved reference retained')
                per_image.append(statistics.median(math.log(max(x,EPS)) if key=='residual_noise_ratio' else math.log(x+EPS) for x in v))
            values.append(statistics.median(per_image))
        return dict(u=u[start:],log_values=values)
    return dict(reference_images=images,GT_used=False,R=curve('residual_noise_ratio',0),Omega=curve('guidance_variation3_per_u',2))


def scores(R,omega,u,ref):
    if not math.isfinite(R) or R<0 or not math.isfinite(u):raise ValueError('Nonfinite physical state')
    def response(value,curve):
        baseline=float(np.interp(u,curve['u'],curve['log_values']))
        return max(-1.,min(1.,(value-baseline)/math.log(2.)))
    e_R=response(math.log(max(R,EPS)),ref['R'])
    if omega is None:e_C=None
    else:
        if not math.isfinite(omega) or omega<0:raise ValueError('Invalid directional state')
        e_C=response(math.log(omega+EPS),ref['Omega'])
    return dict(e_R=e_R,e_C=e_C)


def combined(arm,state):
    if arm not in ARMS:raise ValueError('Unplanned arm')
    if arm=='C':return 0.
    if arm=='R':return state['e_R']
    if arm=='D':return state['e_C'] if state['e_C'] is not None else 0.
    return state['e_R'] if state['e_C'] is None else .5*(state['e_R']+state['e_C'])


class Positions:
    def __init__(self,u_by_t,nominal_descending):
        self.u=np.asarray(u_by_t,dtype=np.float64);self.nominal=list(nominal_descending)
        if len(self.nominal)!=30 or self.nominal[0]!=len(self.u)-1 or self.nominal[-1]!=0 or any(b>=a for a,b in zip(self.nominal,self.nominal[1:])):raise ValueError('Need the exact original 30-point descending grid')
        if not np.isfinite(self.u).all() or abs(self.u[-1])>1e-12 or abs(self.u[0]-1)>1e-12 or not np.all(np.diff(self.u)<0):raise ValueError('Invalid noise coordinate')
        self.base=self.u[self.nominal];self.cells={}
        for j in range(1,29):
            lo=.5*(self.base[j-1]+self.base[j]);hi=.5*(self.base[j]+self.base[j+1])
            options=np.flatnonzero((self.u>=lo)&(self.u<hi))
            if self.nominal[j] not in options or not len(options):raise ValueError('Empty original slot')
            self.cells[j]=options
    def choose(self,j,E,rho=RHO):
        if not 1<=j<=29 or not math.isfinite(E) or not -1<=E<=1 or not 0<=rho<=.25:raise ValueError('Out of frozen response domain')
        if j==29:return 0
        if E==0 or rho==0:return self.nominal[j]
        target=self.base[j]-rho*E*(self.base[j]-self.base[j-1]);candidates=self.cells[j]
        return int(candidates[np.argmin(abs(self.u[candidates]-target))])
    def weights(self,t):
        if t in self.nominal:
            k=self.nominal.index(t);return 29-k,29-k,0.
        value=self.u[t];right=int(np.searchsorted(self.base,value));left=right-1
        if left<0 or right>=30:raise ValueError('Parameter interpolation outside anchors')
        w=float((value-self.base[left])/(self.base[right]-self.base[left]))
        return 29-left,29-right,w


def parameters(z,t,positions):
    i,j,w=positions.weights(t)
    if i==j:return z.zeta[i],z.D[i],dict(left=i,right=j,right_weight=w)
    return (1-w)*z.zeta[i]+w*z.zeta[j],(1-w)*z.D[i]+w*z.D[j],dict(left=i,right=j,right_weight=w)
