"""Inference-only static V mapping cache; current-query gates stay dynamic."""
import torch
import torch.nn.functional as F

from exp.selective_kv.bridge import Injection


class CachedInjection(Injection):
    def __init__(self, controller):
        super().__init__(controller)
        self.static = {}
        self.native_cache = self.selection = None

    def prepare(self, native_cache, selection):
        self.native_cache, self.selection = native_cache, selection

    @torch.no_grad()
    def bind(self, network, positions, source, prior, mode, alpha, shuffle=False):
        if mode != 'gated' or shuffle:
            raise ValueError('Cached inference supports the aligned gated mapper only')
        if self.native_cache is None or self.selection is None:
            raise RuntimeError('Bind requires this round\'s native cache and workset')
        super().bind(network, positions, source, prior, mode, alpha, shuffle)
        self.static = {}
        if positions.numel() and alpha:
            indices = torch.searchsorted(self.selection, positions)
            if not torch.equal(self.selection[indices], positions):
                raise ValueError('Injection positions are outside the selected workset')
            features = torch.stack((prior[positions].clamp_min(1e-12).log()/10,
                                    positions.float()/max(1, prior.shape[-1]-1)), -1)
            for relative, layer in enumerate(network.layers):
                native = self.native_cache[layer][1][0, :, positions].transpose(0, 1).reshape(len(positions), -1).float()
                mapped, latent, scale = network.proposal(relative, source[relative], native)
                sv = network.native[relative](native)
                cosine = F.cosine_similarity(mapped, native, dim=-1)[:, None]
                delta = 2*scale*torch.tanh((mapped-native)/(2*scale))
                self.static[layer] = (relative, indices, native, latent, sv, features, cosine, delta)
        # No ownership of full backing KV beyond bind preparation.
        self.native_cache = self.selection = None

    def clear(self):
        super().clear()
        self.static = {}
        self.native_cache = self.selection = None

    @torch.no_grad()
    def __call__(self, layer, k, v, visible_positions, hidden, query_positions):
        if layer not in self.static:
            return k, v
        if hidden.shape[1] != 1:
            raise ValueError('Current-query gates require sequential Draft queries')
        relative, indices, native, latent, sv, features, cosine, delta = self.static[layer]
        q = self.network.query[relative](hidden[0, -1].float()).expand(len(indices), -1)
        gate_features = torch.cat((latent, sv, q, features, cosine), -1)
        gate = self.network.gates[relative](gate_features).sigmoid()*self.alpha
        corrected = native+gate*delta
        result = v.clone()
        result[0, :, indices, :] = corrected.reshape(len(indices), v.shape[1], v.shape[-1]).transpose(0, 1).to(v.dtype)
        return k, result
