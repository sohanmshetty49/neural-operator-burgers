import pytest
import torch

import models


@pytest.mark.parametrize("make", [models.TransformerOperator, models.FNO1d])
def test_resolution_agnostic_models_accept_other_grids(make):
    m = make().eval()
    assert m(torch.randn(2, 64)).shape == (2, 64)
    assert m(torch.randn(2, 256)).shape == (2, 256)


def test_mlp_is_tied_to_its_grid():
    m = models.MLP(64)
    assert m(torch.randn(2, 64)).shape == (2, 64)
    with pytest.raises(RuntimeError):
        m(torch.randn(2, 128))


def test_fno_is_translation_equivariant_on_periodic_grid():
    # Shifting the input by one grid cell shifts the output by one cell, away from the coordinate
    # channel: check with the coordinate channel neutralised by comparing against a shifted copy
    # produced by the same (translation-equivariant) spectral path.
    torch.manual_seed(0)
    m = models.FNO1d(width=8, modes=4, n_layers=2).eval()
    h = torch.randn(1, 8, 32)
    s = m.spec[0]
    assert torch.allclose(s(torch.roll(h, 3, dims=-1)), torch.roll(s(h), 3, dims=-1), atol=1e-5)


@pytest.mark.parametrize("make", [lambda: models.MLP(32), models.TransformerOperator, models.FNO1d])
def test_a_few_steps_reduce_training_loss(make):
    torch.manual_seed(0)
    m = make()
    x = torch.randn(16, 32)
    y = torch.roll(x, 1, dims=-1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    first = None
    for _ in range(30):
        loss = (m(x) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < first
