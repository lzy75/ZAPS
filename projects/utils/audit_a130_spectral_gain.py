"""CPU-only audit of the A130 transfer function before the denoiser pullback."""
import argparse
from pathlib import Path

import numpy as np

from utils import diag_pigdm_directional_covariance as a130
from utils import paper_clock_alignment as files

STAGES = (('high', 670, 990), ('middle', 330, 669), ('low', 0, 329))


def frequency_groups(size):
    x = np.fft.rfftfreq(size)[None, :]
    y = np.fft.fftfreq(size)[:, None]
    radius = np.sqrt(x*x+y*y)
    radial = np.searchsorted([1/16, 1/8, 1/4], radius, side='left')
    angle = (np.arctan2(y, x)+np.pi/8) % np.pi
    direction = np.minimum(3, (angle/(np.pi/4)).astype(int))
    direction[:, (0, -1)] = 0
    multiplicity = np.full(radius.shape, 2)
    multiplicity[:, (0, -1)] = 1
    return 4*radial+direction, multiplicity


def blur_power(kernel, size):
    kernel = np.asarray(kernel, dtype=np.float32)
    kernel = kernel/np.sum(kernel)
    h, w = kernel.shape
    padded = np.zeros((size, size), np.float32)
    padded[:h, :w] = kernel
    # The real circular shift of the PSF changes phase, leaving this power invariant.
    return np.square(np.abs(np.fft.rfft2(padded)))


def audit_step(power, lam, variance_by_band, bands, mult):
    power = power.astype(np.float64, copy=False)
    cov = variance_by_band.astype(np.float64, copy=False)[bands]
    native = power/(power+lam)**2
    changed = power/(power*cov+lam)**2
    ratio = (power+lam)/(power*cov+lam)
    excess = mult*np.maximum(0., changed-native)
    by_band = np.bincount(bands.ravel(), weights=excess.ravel(), minlength=16)
    band = int(np.argmax(by_band))
    peak = np.unravel_index(np.argmax(ratio), ratio.shape)
    return dict(max_amplitude_gain=float(ratio[peak]), peak_band=int(bands[peak]),
                peak_frequency=[float(np.fft.fftfreq(power.shape[0])[peak[0]]),
                                float(np.fft.rfftfreq(power.shape[0])[peak[1]])],
                white_residual_rms_gain=float(np.sqrt(np.sum(changed*mult)/np.sum(native*mult))),
                fraction_frequencies_gain_above_2=float(mult[ratio > 2].sum()/mult.sum()),
                excess_energy_dominant_band=band if by_band.sum() > 0 else None,
                excess_energy_dominant_band_share=float(by_band[band]/by_band.sum()) if by_band.sum() > 0 else 0.)


def residual_trace(root, cases):
    schedules = []
    for case in cases:
        path = root/'cases'/case['key']/'SCALAR'
        row = files.read(path/'run.json')
        trace_path = path/'trace.json'
        files.require(row['status'] == 'complete' and files.digest(trace_path) == row['artifacts']['trace.json'],
                      'Archived scalar trace is incomplete or changed')
        schedule = {step['t']:step['lambda_noise'] for step in files.read(trace_path)}
        files.require(tuple(schedule) == a130.GRID and all(np.isfinite(v) and v > 0 for v in schedule.values()),
                      'Native regularizer schedule differs')
        schedules.append(schedule)
    files.require(all(s == schedules[0] for s in schedules[1:]), 'Expected identical measurement noise/grid')
    return schedules[0]


def execute(root, output=None):
    summary = files.read(root/'summary.json')
    files.require(summary['experiment'] == 'A130' and summary['status'] == 'complete',
                  'Completed A130 required')
    config = files.read(root/'resolved_config.json')
    profile = files.read(root/'covariance_profile.json')
    # A130 identity includes the resolved A124R1 plan, which owns the actual motion PSF.
    paths = [Path(name) for name in config['sources'] if
             '/diag_a118_blur_transfer/A124R1_pigdm_ffhq/' in name and Path(name).name == 'plan.json']
    files.require(len(paths) == 1 and files.digest(paths[0]) == config['sources'][str(paths[0])],
                  'A124R1 operator plan changed')
    source = files.read(paths[0])
    motion_path = source['motion_kernel']
    files.require(files.digest(motion_path) == source['motion_kernel_sha256'], 'A130 motion PSF changed')
    motion = np.load(motion_path, allow_pickle=False)
    axis = np.arange(-30, 31, dtype=np.float64)
    gaussian = np.exp(-.5*(axis/3)**2)
    kernels = {'gaussian_deblur':np.outer(gaussian, gaussian), 'motion_deblur':motion}
    bands, multiplicity = frequency_groups(a130.SIZE)
    mass = np.bincount(bands.ravel(), weights=multiplicity.ravel(), minlength=16)
    files.require(np.array_equal(mass, profile['band_sizes']), 'A130 frequency groups differ from saved profile')
    rows = []
    for task, kernel in kernels.items():
        power = blur_power(kernel, a130.SIZE)
        cases = [case for case in config['cases'] if case['task'] == task]
        schedule = residual_trace(root, cases)
        for t, lam in schedule.items():
            variance = np.asarray(profile['normalized_variance'][str(t)], dtype=np.float64)
            files.require(variance.shape == (16,) and np.isfinite(variance).all() and
                          (variance > 0).all() and abs(np.dot(variance, mass)/mass.sum()-1) < 1e-10,
                          'Saved normalized covariance is invalid')
            rows.append(dict(task=task, t=t, lambda_noise=lam,
                **audit_step(power, lam, variance.astype(np.float32), bands, multiplicity)))
        for stage, low, high in STAGES:
            phase = [row for row in rows if row['task'] == task and low <= row['t'] <= high]
            peak = max(phase, key=lambda row:row['max_amplitude_gain'])
            broad = max(phase, key=lambda row:row['white_residual_rms_gain'])
            print('A130 SPECTRAL', task, stage,
                  dict(peak_ratio=peak['max_amplitude_gain'], peak_t=peak['t'],
                       peak_band=peak['peak_band'], peak_frequency=peak['peak_frequency'],
                       rms_ratio=broad['white_residual_rms_gain'], rms_t=broad['t'],
                       fraction_gain_above_2=broad['fraction_frequencies_gain_above_2'],
                       dominant_excess_band=broad['excess_energy_dominant_band'],
                       dominant_excess_share=broad['excess_energy_dominant_band_share']), flush=True)
    if output:
        if Path(output).exists():
            raise FileExistsError(output)
        files.write(output, dict(nn_forwards=0, rows=rows,
            scope='FFT linear operator before denoiser pullback or scalar. White residual RMS is a probe, not measured residual energy.'))
    print('A130 SPECTRAL COMPLETE', 'nn_forwards=0', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=a130.ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    execute(args.root, args.output)


if __name__ == '__main__':
    main()
