import decimal
import unittest
import numpy as np
from scipy import sparse
from Design.SMRF_Accurate_Sparse_Action import accurate_sparse_action


def oracle(a, x):
    with decimal.localcontext() as ctx:
        ctx.prec = 800
        D = decimal.Decimal.from_float
        return np.asarray([float(sum((D(float(a.data[j])) * D(float(x[a.indices[j]]))
            for j in range(a.indptr[i], a.indptr[i + 1])), decimal.Decimal(0)))
            for i in range(a.shape[0])])


class AccurateSparseAction(unittest.TestCase):
    def test_product_cancellation_needs_both_parts(self):
        a = sparse.csr_matrix([[1.+2.**-27, -1.], [1e15, 1.]])
        x = np.array([1.-2.**-27, 1.])
        got = accurate_sparse_action(a, x)
        np.testing.assert_array_equal(got, oracle(a, x))
        self.assertEqual(got[0], -2.**-54)
        self.assertEqual((a@x)[0], 0.)

    def test_independent_decimal_random_and_domain_extremes(self):
        rng = np.random.default_rng(215)
        for exp in (0, -450, 450):
            # Powers at the exact allowed endpoints and nearby binary values.
            value = 2.**exp
            a = sparse.csr_matrix([[value, -value, value], [0., value, 0.]])
            x = np.array([value, value, value])
            np.testing.assert_array_equal(accurate_sparse_action(a, x), oracle(a, x))
        a = sparse.csr_matrix(rng.normal(size=(24, 31))*10.**rng.integers(-6, 7, size=(24, 31)))
        x = rng.normal(size=31)
        np.testing.assert_allclose(accurate_sparse_action(a, x), oracle(a, x), rtol=2e-16, atol=0.)

    def test_duplicate_rows_empty_rows_and_order(self):
        for values in ([1e16, 1., -1e16], [-1e16, 1e16, 1.], [1., -1e16, 1e16]):
            a = sparse.csr_matrix((values, [0, 0, 0], [0, 3, 3]), shape=(2, 3))
            np.testing.assert_array_equal(accurate_sparse_action(a, [1., 2., 3.]), [1., 0.])
        a = sparse.csr_matrix((16386, 2))
        np.testing.assert_array_equal(accurate_sparse_action(a, [0., 0.]), np.zeros(16386))

    def test_invalid_inputs_and_outside_verified_domain_refused(self):
        for a, x in ((np.eye(2), [1., 1.]), (sparse.coo_matrix(np.eye(2)), [1., 1.]),
                     (sparse.eye(2, format='csr'), [1.]), (sparse.csr_matrix([[np.nan]]), [1.]),
                     (sparse.csr_matrix([[1.]]), [np.inf]), (sparse.csr_matrix([[1j]]), [1.])):
            with self.assertRaises(ValueError): accurate_sparse_action(a, x)
        for a, x in ((2.**-451, 1.), (1., 2.**451), (1e308, 1e-308)):
            with self.assertRaises(FloatingPointError): accurate_sparse_action(sparse.csr_matrix([[a]]), [x])
        np.testing.assert_array_equal(accurate_sparse_action(sparse.csr_matrix([[1e308]]), [0.]), [0.])


if __name__ == '__main__': unittest.main()
