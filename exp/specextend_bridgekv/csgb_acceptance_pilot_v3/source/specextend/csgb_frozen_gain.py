"""Frozen training-state gain predictor; deployment never fits or updates."""
import numpy as np


class FrozenGain:
    def __init__(self, state):
        self.budgets=tuple(state['budgets']); self.cols=tuple(state['cols'])
        self.mean=np.array(state['mean'],dtype=float); self.scale=np.array(state['scale'],dtype=float)
        self.theta=np.array(state['theta'],dtype=float)
        if self.theta.shape != (len(self.cols),len(self.budgets)):
            raise ValueError('Invalid frozen coefficient shape')
        if (not self.cols or self.cols[0]!=0 or len(set(self.cols))!=len(self.cols)
                or any(i not in range(4) for i in self.cols)
                or self.budgets!=(8,16,24,32)
                or self.mean.shape!=(len(self.cols)-1,) or self.scale.shape!=self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not np.isfinite(self.theta).all() or np.any(self.scale<=0)):
            raise ValueError('Invalid frozen parameters')
        for a in (self.mean,self.scale,self.theta): a.setflags(write=False)

    def choose(self,x,explore=False):
        if explore: raise ValueError('Frozen evaluation cannot explore')
        if x is None: return 8
        x=np.asarray(x,dtype=float)
        if x.shape!=(4,): raise ValueError('Expected four CSGB features')
        z=x[list(self.cols)].copy()
        if not np.isfinite(z).all(): return 8
        z[1:]=(z[1:]-self.mean)/self.scale
        return self.budgets[int(np.argmax(z@self.theta))]
