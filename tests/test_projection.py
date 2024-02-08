import sys
sys.path.append('../')
from projection import calc_G
import jax.numpy as jnp
import pytest

@pytest.mark.parametrize(
    ('input_n', 'expected'),
    (
        (jnp.array([[1.], [1.]]), jnp.array([[8., 14.],[14., 25.]])),
    )
)


def test_calc_G(input_n, expected):
    def featurize(xx):
        return jnp.concatenate([xx[0]**2 + xx[1]**2, 3 * xx[0] + 4 * xx[1]])
    ## Calculate G
    _, G = calc_G(input_n, featurize)
    ## Compare G to solution
    assert jnp.all(G == expected), print(input_n.shape)