import jax.numpy as jnp
import numpy as np

from rmsd_map.rmsd.kabsch_jax import kabsch_full_improp, kabsch_full_proper


def test_proper_rotation_for_planar_mirror():
    a = jnp.array([[[-1., -1., 0.], [1., -1., 0.],
                    [-1.,  1., 0.], [1.,  1., 0.]]])
    b = a * jnp.array([-1., 1., 1.])
    result = kabsch_full_proper(a, b, jnp.arange(4)[None, :])
    q = np.asarray(result["Rotation"][0])
    np.testing.assert_allclose(np.linalg.det(q), 1.0, atol=1e-5)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(result["Transformed"], b, atol=1e-5)
    np.testing.assert_allclose(result["RMSD"], 0.0, atol=1e-5)


def test_proper_rotation_cannot_match_nonplanar_mirror():
    a = jnp.array([[[1., 2., 3.], [1., -2., -3.],
                    [-1., 2., -3.], [-1., -2., 3.]]])
    b = a * jnp.array([-1., 1., 1.])
    perm = jnp.arange(4)[None, :]
    proper = kabsch_full_proper(a, b, perm)
    improper = kabsch_full_improp(a, b, perm)
    np.testing.assert_allclose(proper["RMSD"], 2.0, atol=1e-5)
    np.testing.assert_allclose(improper["RMSD"], 0.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(proper["Rotation"]), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(improper["Rotation"]), -1.0, atol=1e-5)
    q = np.asarray(proper["Rotation"][0])
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(a @ q, proper["Transformed"], atol=1e-5)
    residual = proper["Transformed"] - b
    np.testing.assert_allclose(jnp.sqrt(jnp.mean(jnp.sum(residual**2, axis=-1), axis=1)),
                               proper["RMSD"], atol=1e-5)
