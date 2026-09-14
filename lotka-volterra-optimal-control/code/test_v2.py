"""Run with pytest -q. These checks do not use the optimizer to verify derivatives."""
import numpy as np
import pytest
from lotka_v2 import Problem,optimize

@pytest.mark.parametrize('scheme',['imex','positive','flux'])
def test_derivatives(scheme):
    P=Problem(M=12,N=20,scheme='positive' if scheme=='flux' else scheme,harvest_rule='flux' if scheme=='flux' else 'left'); rng=np.random.default_rng(17)
    u=np.full(P.shape,.7); d=rng.normal(size=P.shape); e=rng.normal(size=P.shape)
    f,g,v,p=P.evaluate(u); Hd=P.hessian(u,d,v,p); He=P.hessian(u,e,v,p); eps=2e-5
    fd=(P.objective(u+eps*d)-P.objective(u-eps*d))/(2*eps)
    assert abs(fd-P.inner(g,d))<2e-7
    gd=(P.gradient(u+eps*d)-P.gradient(u-eps*d))/(2*eps)
    assert P.norm(gd-Hd)<1e-7
    assert abs(P.inner(Hd,e)-P.inner(d,He))<1e-11

@pytest.mark.parametrize('N',[1,2,4,6,8,20,80])
def test_positive_balance(N):
    P=Problem(M=24,N=N,scheme='positive'); u=np.full(P.shape,P.par.umax)
    _,g,v,p=P.evaluate(u); a=P.diagnostics(u,v,p,g)
    assert a['min_state']>=0
    assert a['balance_error']<2e-11

@pytest.mark.parametrize('scheme',['imex','positive','flux'])
def test_stationary_control(scheme):
    P=Problem(M=12,N=30,scheme='positive' if scheme=='flux' else scheme,harvest_rule='flux' if scheme=='flux' else 'left'); u,r=optimize(P,tol=2e-7)
    assert r['residual']<1e-6
    assert np.all(u>=0) and np.all(u<=P.par.umax)
