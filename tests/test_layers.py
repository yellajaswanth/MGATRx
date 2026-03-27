import torch
import pytest

from source.layers import GraphConvolution, CosineGraphAttentionLayer, DictReLU, DictDropout


N, D_IN, D_OUT = 8, 16, 32


class TestGraphConvolution:
    def setup_method(self):
        self.layer = GraphConvolution(D_IN, D_OUT)
        self.x = torch.randn(N, D_IN)
        self.adj = torch.eye(N)

    def test_output_shape_with_adj(self):
        out = self.layer(self.x, self.adj)
        assert out.shape == (N, D_OUT)

    def test_output_shape_scalar_adj(self):
        out = self.layer(self.x, adj=1.0)
        assert out.shape == (N, D_OUT)

    def test_output_shape_int_adj(self):
        out = self.layer(self.x, adj=2)
        assert out.shape == (N, D_OUT)

    def test_no_nan_in_output(self):
        out = self.layer(self.x, self.adj)
        assert not torch.isnan(out).any()

    def test_sparse_input_accepted(self):
        x_sparse = self.x.to_sparse()
        out = self.layer(x_sparse, adj=1.0)
        assert out.shape == (N, D_OUT)


class TestCosineGraphAttentionLayer:
    def setup_method(self):
        self.layer = CosineGraphAttentionLayer(requires_grad=True)
        self.xi = torch.randn(N, D_IN)
        self.xj = torch.randn(N, D_IN)
        self.adj = torch.eye(N)

    def test_output_shape_with_adj(self):
        out = self.layer(self.xi, self.xj, self.adj)
        assert out.shape == (N, D_IN)

    def test_output_shape_scalar_adj(self):
        out = self.layer(self.xi, self.xj, adj=1.0)
        assert out.shape == (N, D_IN)

    def test_no_nan_in_output(self):
        out = self.layer(self.xi, self.xj, self.adj)
        assert not torch.isnan(out).any()

    def test_beta_is_learnable(self):
        assert self.layer.beta.requires_grad is True

    def test_device_of_output_matches_input(self):
        out = self.layer(self.xi, self.xj, adj=1.0)
        assert out.device == self.xi.device


class TestDictReLU:
    def test_dict_input(self):
        relu = DictReLU()
        x = {0: torch.randn(4, 8), 1: torch.randn(4, 8)}
        out = relu(x)
        assert set(out.keys()) == {0, 1}
        assert (out[0] >= 0).all()

    def test_tensor_input(self):
        relu = DictReLU()
        x = torch.randn(4, 8)
        out = relu(x)
        assert (out >= 0).all()
        assert out.shape == x.shape


class TestDictDropout:
    def test_dict_input_eval_mode(self):
        drop = DictDropout(p=0.5)
        drop.eval()
        x = {0: torch.ones(4, 8)}
        out = drop(x)
        assert torch.allclose(out[0], x[0])

    def test_tensor_input_eval_mode(self):
        drop = DictDropout(p=0.5)
        drop.eval()
        x = torch.ones(4, 8)
        out = drop(x)
        assert torch.allclose(out, x)
