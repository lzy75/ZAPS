"""Required real-Torch gates for A102 shared optimizer/RNG replay."""
import copy, unittest


class TorchTests(unittest.TestCase):
    def test_adam_state_and_rng_replay_are_exact(self):
        import torch
        torch.manual_seed(7)
        first = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.Adam([first], lr=.01)
        (first.square().sum()).backward(); optimizer.step(); optimizer.zero_grad()
        state = copy.deepcopy(optimizer.state_dict()); value = first.detach().clone()
        rng = torch.get_rng_state().clone()
        results=[]
        for _ in range(2):
            parameter=torch.nn.Parameter(value.clone()); candidate=torch.optim.Adam([parameter],lr=.01)
            candidate.load_state_dict(copy.deepcopy(state));torch.set_rng_state(rng)
            noise=torch.randn(4);(parameter.square().sum()).backward();candidate.step()
            results.append((parameter.detach().clone(),noise,torch.get_rng_state().clone()))
        for a,b in zip(results[0],results[1]):torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
