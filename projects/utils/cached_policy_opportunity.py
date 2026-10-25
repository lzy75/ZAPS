"""Finite-library, post-hoc policy opportunity bounds; never an online controller."""
import numpy as np

ARMS=('C','TIME','OLD','REG','BLIND','TIME_NOISE','UP_REG','DOWN_REG')
METRICS=('psnr','ssim','lpips','loss')


def admissible(values):
    return (values[...,1]>=-.002)&(values[...,2]<=.005)


def analyze(values,arms=ARMS):
    """values[image, policy, metric] are deltas from C within ONE noise condition."""
    values=np.asarray(values,dtype=float)
    if (values.ndim!=3 or values.shape[1:]!=(len(arms),4) or 'C' not in arms
            or values.shape[0]<2 or not np.isfinite(values).all()
            or len(set(arms))!=len(arms) or len(arms)**values.shape[0]>1000000):
        raise ValueError('Invalid or oversized finite policy library')
    n,k,_=values.shape
    if not np.array_equal(values[:,arms.index('C')],np.zeros((n,4))):
        raise ValueError('C must be exact zero reference')
    means=values.mean(axis=0)
    def summarize(choices):
        selected=values[np.arange(n),choices];mean=selected.mean(axis=0)
        return dict(actions=[arms[i] for i in choices],mean_delta=dict(zip(METRICS,mean.tolist())),
                    quality_guard=bool(admissible(mean)),
                    images_psnr_positive=int((selected[:,0]>1e-9).sum()),
                    max_image_lpips_delta=float(selected[:,2].max()),
                    min_image_psnr_delta=float(selected[:,0].min()))
    feasible=admissible(means)
    fixed=int(np.argmax(np.where(feasible,means[:,0],-np.inf)))
    best_fixed=summarize(np.full(n,fixed,dtype=int))
    unguarded_fixed=int(np.argmax(means[:,0]))
    # Exact enumeration is small (8^6); this is a cohort-coordinated GT diagnostic bound.
    combinations=np.indices((k,)*n,dtype=np.int16).reshape(n,-1).T
    pooled=np.zeros((len(combinations),4),dtype=float)
    for i in range(n):pooled+=values[i,combinations[:,i]]
    pooled/=n
    allowed=admissible(pooled)
    chosen=int(np.argmax(np.where(allowed,pooled[:,0],-np.inf)))
    constrained=summarize(combinations[chosen])
    unrestricted=summarize(np.argmax(values[:,:,0],axis=1))
    individual=np.argmax(np.where(admissible(values),values[:,:,0],-np.inf),axis=1)
    # Leave out all outcomes of the held image, but use GT outcomes from the other images.
    # This is exploratory noise-only calibration, not prospective independent validation.
    loo=[]
    for i in range(n):
        train=np.delete(values,i,axis=0).mean(axis=0)
        loo.append(int(np.argmax(np.where(admissible(train),train[:,0],-np.inf))))
    return dict(image_clusters=n,policy_count=k,enumerated_combinations=len(combinations),
                best_fixed_feasible=best_fixed,best_fixed_unguarded=summarize(np.full(n,unguarded_fixed,dtype=int)),
                GT_cohort_mean_guard_oracle=constrained,
                GT_individual_guard_oracle=summarize(individual),GT_unguarded_oracle=unrestricted,
                leave_one_image_out_noise_only=summarize(np.asarray(loo)),
                oracle_headroom_over_best_fixed=constrained['mean_delta']['psnr']-best_fixed['mean_delta']['psnr'],
                feasible_fixed_arms=[arms[i] for i in np.flatnonzero(feasible)],
                limits='All outcomes previously observed; GT oracle is not deployable. Cohort-mean oracle coordinates '
                       'all images using GT and may offset one image risk with another. Individual-guard oracle is an '
                       'additional stricter descriptive bound, not a replacement for the preregistered mean guard. '
                       'Leave-one-image-out selects a full-path policy using only other images, not target states. '
                       'No trajectory-local action, state-predictor value, generalization or efficacy claim.')
