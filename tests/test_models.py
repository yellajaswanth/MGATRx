import torch
import pytest

from source.models import MGATRx, HeteroGCN, HeteroGAT


N_DRUG, N_DIS, N_TARGET = 10, 8, 6
D_IN = {0: N_DRUG, 1: N_DIS, 2: N_TARGET}
D_OUT = 16
TASKS = ((0, 1),)


def make_fea_mats():
    return {k: torch.eye(v) for k, v in D_IN.items()}


def make_adj_mats():
    return {
        (0, 1): [torch.randint(0, 2, (N_DRUG, N_DIS)).float()],
        (0, 2): [torch.randint(0, 2, (N_DRUG, N_TARGET)).float()],
    }


class TestHeteroGCN:
    def test_output_keys_match_input(self):
        layer = HeteroGCN(D_IN, D_OUT)
        fea = make_fea_mats()
        adj = make_adj_mats()
        out = layer(fea, adj)
        assert set(out.keys()) == set(D_IN.keys())

    def test_output_shape(self):
        layer = HeteroGCN(D_IN, D_OUT)
        fea = make_fea_mats()
        adj = make_adj_mats()
        out = layer(fea, adj)
        assert out[0].shape == (N_DRUG, D_OUT)
        assert out[1].shape == (N_DIS, D_OUT)

    def test_no_nan(self):
        layer = HeteroGCN(D_IN, D_OUT)
        fea = make_fea_mats()
        adj = make_adj_mats()
        out = layer(fea, adj)
        for v in out.values():
            assert not torch.isnan(v).any()

    def test_invalid_in_dim_raises(self):
        with pytest.raises(ValueError):
            HeteroGCN(in_dim="bad", out_dim=16)


class TestHeteroGAT:
    def test_output_keys_match_input(self):
        layer = HeteroGAT(D_IN, D_OUT)
        fea = make_fea_mats()
        adj = make_adj_mats()
        out = layer(fea, adj)
        assert set(out.keys()) == set(D_IN.keys())

    def test_output_shape(self):
        layer = HeteroGAT(D_IN, D_OUT)
        fea = make_fea_mats()
        adj = make_adj_mats()
        out = layer(fea, adj)
        assert out[0].shape == (N_DRUG, D_OUT)

    def test_int_in_dim_raises(self):
        with pytest.raises(NotImplementedError):
            HeteroGAT(in_dim=16, out_dim=16)


class TestMGATRxGCN:
    def setup_method(self):
        self.model = MGATRx(D_IN, (D_OUT,), tasks=TASKS, model='GCN')
        self.fea = make_fea_mats()
        self.adj = make_adj_mats()

    def test_forward_returns_recon_and_embeddings(self):
        recon, z = self.model(self.fea, self.adj, None)
        assert isinstance(recon, dict)
        assert isinstance(z, dict)

    def test_recon_shape(self):
        recon, _ = self.model(self.fea, self.adj, None)
        assert recon[(0, 1)][0].shape == (N_DRUG, N_DIS)

    def test_embedding_shape(self):
        _, z = self.model(self.fea, self.adj, None)
        assert z[0].shape == (N_DRUG, D_OUT)

    def test_no_nan_in_recon(self):
        recon, _ = self.model(self.fea, self.adj, None)
        assert not torch.isnan(recon[(0, 1)][0]).any()

    def test_gradients_flow(self):
        recon, _ = self.model(self.fea, self.adj, None)
        loss = recon[(0, 1)][0].sum()
        loss.backward()
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()


class TestMGATRxGAT:
    def setup_method(self):
        self.model = MGATRx(D_IN, (D_OUT,), tasks=TASKS, model='GAT')
        self.fea = make_fea_mats()
        self.adj = make_adj_mats()

    def test_forward_recon_shape(self):
        recon, _ = self.model(self.fea, self.adj, None)
        assert recon[(0, 1)][0].shape == (N_DRUG, N_DIS)

    def test_no_nan_in_recon(self):
        recon, _ = self.model(self.fea, self.adj, None)
        assert not torch.isnan(recon[(0, 1)][0]).any()
