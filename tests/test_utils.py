import numpy as np
import scipy.sparse as sp
import torch
import pytest

from source.utils import normalize, sparse_to_tensor, to_sparse, to_dense, issymmetric


class TestToDenseToSparse:
    def test_to_dense_already_dense(self):
        t = torch.randn(3, 4)
        assert to_dense(t) is t

    def test_to_dense_from_sparse(self):
        t = torch.eye(4).to_sparse()
        dense = to_dense(t)
        assert not dense.is_sparse
        assert dense.shape == (4, 4)

    def test_to_sparse_already_sparse(self):
        t = torch.eye(3).to_sparse()
        assert to_sparse(t) is t

    def test_to_sparse_from_dense(self):
        t = torch.eye(3)
        sparse = to_sparse(t)
        assert sparse.is_sparse

    def test_round_trip(self):
        t = torch.randint(0, 2, (5, 5)).float()
        assert torch.allclose(to_dense(to_sparse(t)), t)


class TestSparseToTensor:
    def test_shape_preserved(self):
        mat = sp.eye(6, format='csr')
        tensor = sparse_to_tensor(mat)
        assert tensor.shape == torch.Size([6, 6])
        assert tensor.is_sparse

    def test_values_correct(self):
        mat = sp.coo_matrix(np.eye(4))
        tensor = sparse_to_tensor(mat)
        dense = tensor.to_dense()
        assert torch.allclose(dense, torch.eye(4))

    def test_non_coo_input(self):
        mat = sp.csr_matrix(np.eye(3))
        tensor = sparse_to_tensor(mat)
        assert tensor.shape == torch.Size([3, 3])


class TestNormalize:
    def test_asymmetric_returns_tensor(self):
        t = torch.ones(3, 3)
        result = normalize(t, issymmetric=False)
        assert torch.is_tensor(result)

    def test_asymmetric_row_sums_to_one(self):
        data = torch.FloatTensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        result = normalize(data, issymmetric=False).to_dense()
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)

    def test_symmetric_returns_array(self):
        t = torch.eye(4)
        result = normalize(t, issymmetric=True)
        assert isinstance(result, (np.ndarray, np.matrix))


class TestIsSymmetric:
    def test_symmetric_tensor(self):
        t = torch.eye(4)
        assert issymmetric(t) is True

    def test_asymmetric_tensor(self):
        t = torch.FloatTensor([[0, 1], [0, 0]])
        assert issymmetric(t) is False

    def test_symmetric_numpy(self):
        a = np.array([[1, 2], [2, 1]])
        assert issymmetric(a) is True

    def test_non_square(self):
        a = np.ones((2, 3))
        assert issymmetric(a) is False

    def test_sparse_symmetric(self):
        mat = sp.eye(5, format='csr')
        assert issymmetric(mat) is True
