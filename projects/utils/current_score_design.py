"""A50 immutable cohort/phase design and label-free control allocation."""
import math
import statistics

TASKS = ("super_resolution", "gaussian_deblur", "inpainting", "motion_deblur")
IMAGES = ("00446", "00559", "00608", "00873", "00951")
GENERATE_ARMS = ("C_NATIVE", "C_IEEE", "PROX_Q", "DIRECT_Q", "ALIGN_Q")
CONTROL_ARMS = ("TIME", "CONSTANT", "SHUFFLE")
METROLOGY_STEPS = (8, 24, 28)
METROLOGY_CALLS = 6*len(METROLOGY_STEPS)
PHASES = ("generate", "control")


def phase_limit(case, phase):
    if phase == "generate":
        return 30*len(GENERATE_ARMS)+(METROLOGY_CALLS if case["image_id"]==IMAGES[0] else 0)
    if phase == "control":
        return 30*len(CONTROL_ARMS)
    raise ValueError("Unknown phase")


def allocate(cases, curves):
    """Other-image stage and constant controls; one prespecified cyclic donor."""
    if len(cases)!=20 or set(curves)!={c["key"] for c in cases}:
        raise ValueError("Need20 complete unlabeled trajectories")
    if len({c["key"] for c in cases})!=20:
        raise ValueError("Duplicate case")
    result={}
    for task in TASKS:
        group=sorted((c for c in cases if c["task"]==task),key=lambda c:c["image_id"])
        if [c["image_id"] for c in group]!=list(IMAGES):
            raise ValueError("Frozen five-image/four-task cohort differs")
        for c in group:
            values=curves[c["key"]]
            if len(values)!=30 or any(not math.isfinite(v) or v<0. for v in values):
                raise ValueError("Need30 finite nonnegative actions, not labels")
        for i,c in enumerate(group):
            others=[o for o in group if o["image_id"]!=c["image_id"]]
            donor=group[(i+1)%len(group)]
            stage=[float(statistics.median(curves[o["key"]][k] for o in others)) for k in range(30)]
            constant=float(statistics.median(v for o in others for v in curves[o["key"]]))
            result[c["key"]]=dict(TIME=stage,CONSTANT=[constant]*30,SHUFFLE=list(curves[donor["key"]]),
                                  reference_images=[o["image_id"] for o in others],donor=donor["key"])
    return dict(assignments=result,GT_used=False,
                scope="Development-only, whole-trajectory controls. Current target image excluded from its time/constant references.")


def expected_forward_trace(case, grid, phase):
    times=list(reversed(grid))
    if len(times)!=30 or times[0]!=999 or times[-1]!=0:
        raise ValueError("Need frozen30-call grid")
    if phase=="control":
        return times*len(CONTROL_ARMS)
    if phase!="generate":
        raise ValueError("Unknown phase")
    result=[]
    for name in GENERATE_ARMS:
        for k,t in enumerate(times):
            result.append(t)
            if name=="C_IEEE" and case["image_id"]==IMAGES[0] and k in METROLOGY_STEPS:
                result.extend([t]*6)
    return result
