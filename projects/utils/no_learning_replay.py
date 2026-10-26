"""A57 independent RNG recipe; no optimized parameters or network execution."""


def burn_count(reference_epochs=10, steps=30):
    if not isinstance(reference_epochs, int) or reference_epochs < 1 or steps != 30:
        raise ValueError('Need original30-point protocol and positive reference rounds')
    return (reference_epochs-1)*(steps-1)


def noise_from_seed(seed, device, shape=(1, 3, 256, 256), reference_epochs=10):
    import torch
    from utils.diag_learning_rate_ablation import set_seed
    from utils.diag_zaps_sampler_parity import rng_state
    set_seed(seed)
    initial = torch.randn(*shape, device=device)
    for _ in range(burn_count(reference_epochs)):
        torch.randn_like(initial)
    before = rng_state(device)
    innovations = [torch.randn_like(initial) for _ in range(29)]
    return initial, innovations, before
