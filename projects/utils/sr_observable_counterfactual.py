"""A99 exact separable-H row/null final-image counterfactuals, no diffusion calls."""
import numpy as np
ARMS=('SCALAR_FINAL','DIAG_FINAL','ROW_ONLY_DIFFERENCE','NULL_ONLY_DIFFERENCE','LL2_ONLY_DIFFERENCE')


def matrix_from_taps(indices,weights,length):
    indices=np.asarray(indices,dtype=np.int64);weights=np.asarray(weights,dtype=np.float64)
    if indices.shape!=weights.shape or indices.ndim!=2:raise ValueError('Expected native tap x output matrices')
    if not np.isfinite(weights).all() or indices.min()<0 or indices.max()>=length:raise ValueError('Invalid native resize taps')
    matrix=np.zeros((indices.shape[1],length),dtype=np.float64)
    # Reflection boundaries can repeat an input index; accumulation is essential.
    rows=np.broadcast_to(np.arange(indices.shape[1]),indices.shape)
    np.add.at(matrix,(rows.reshape(-1),indices.reshape(-1)),weights.reshape(-1))
    return matrix


def native_matrices(resizer,shape):
    result={}
    for dim,fov,weight in zip(resizer.sorted_dims,resizer.field_of_view,resizer.weights):
        if dim not in (2,3):raise ValueError('Only native spatial separable4x resizing is registered')
        indices=fov.detach().cpu().numpy();weights=weight.detach().cpu().numpy().reshape(indices.shape)
        result[dim]=matrix_from_taps(indices,weights,shape[dim])
    if set(result)!={2,3}:raise ValueError('Missing separable spatial axis')
    return result[2],result[3]


def forward(x,ay,ax):return ay@np.asarray(x,dtype=np.float64)@ax.T


class Projector:
    def __init__(self,ay,ax):
        self.ay=np.asarray(ay,dtype=np.float64);self.ax=np.asarray(ax,dtype=np.float64);self.axes=[];self.info=[]
        for a in (self.ay,self.ax):
            _,s,vh=np.linalg.svd(a,full_matrices=False);tol=np.finfo(np.float64).eps*max(a.shape)*s[0]
            if int((s>tol).sum())!=a.shape[0]:raise ValueError('Native resize is not full row rank; no numerical-policy substitution')
            self.axes.append(vh.T);self.info.append(dict(rows=a.shape[0],columns=a.shape[1],rank=len(s),condition=float(s[0]/s[-1])))
    def row(self,x):
        qy,qx=self.axes;return qy@(qy.T@np.asarray(x,dtype=np.float64)@qx)@qx.T
    def null(self,x):return np.asarray(x,dtype=np.float64)-self.row(x)
    def H(self,x):return forward(x,self.ay,self.ax)


def counterfactuals(scalar,diagonal,projector):
    s=np.asarray(scalar,dtype=np.float64);d=np.asarray(diagonal,dtype=np.float64);difference=d-s
    observed=projector.row(difference);missing=difference-observed
    return dict(SCALAR_FINAL=s,DIAG_FINAL=d,ROW_ONLY_DIFFERENCE=s+observed,NULL_ONLY_DIFFERENCE=s+missing),observed,missing


def relative(a,b):return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-30))


def risks(image,truth,projector):
    error=np.asarray(image,dtype=np.float64)-truth;row=projector.row(error);null=error-row
    result=dict(raw_mse=float(np.mean(error**2)),row_mse=float(np.mean(row**2)),null_mse=float(np.mean(null**2)))
    if abs(result['raw_mse']-result['row_mse']-result['null_mse'])>1e-10*max(result['raw_mse'],1e-12):raise RuntimeError('Exact-H orthogonal error identity failed')
    result.update(clamped_mse=float(np.mean((np.clip(image,-1,1)-truth)**2)),clamp_fraction=float(np.mean(np.abs(image)>1)))
    return result


def decision(rows):
    excess=sum(r['raw_excess'] for r in rows);null=sum(r['null_excess'] for r in rows)
    ratio=null/excess if excess>0 else None
    improved=sum(r['row_only_raw_improvement_over_diag']>0 for r in rows)
    return dict(images=len(rows),sum_raw_excess=excess,sum_null_excess=null,null_share_of_net_excess=ratio,
        row_only_raw_improved_images=improved,supports_terminal_null_risk_contribution=len(rows)==3 and excess>0 and null>0 and ratio>.5 and improved>=2,
        scope='Terminal same-source counterfactual only. Ratio can exceed1 when observable error improves. Does not prove high-dimensional optimization caused drift, nor that an online constrained policy improves.')
