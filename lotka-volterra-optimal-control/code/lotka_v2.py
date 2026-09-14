"""Standalone reproducibility code for the accompanying Lotka--Volterra article v2.

Two state schemes, their exact weighted gradients and Hessian-vector products.
Tracking and effort use left endpoints; the positive state scheme supports
both old-state and flux-aligned harvest valuation. No state clipping or
finite-difference adjoint is used. This file implements the manuscript equations; it is not a downloaded
copy of the author's public repository.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, eigsh

Array = NDArray[np.float64]
Scheme = Literal['imex', 'positive']

@njit(cache=True)
def _solve(diff, loss, rhs):
    """Thomas solve of I - dt*mu*Delta_h + diag(loss), Neumann ghost nodes."""
    size = rhs.size
    c = np.empty(size); y = np.empty(size)
    pivot = 1.0 + 2.0*diff + loss[0]
    c[0] = -2.0*diff/pivot; y[0] = rhs[0]/pivot
    for j in range(1, size):
        lo = -2.0*diff if j == size-1 else -diff
        up = 0.0 if j == size-1 else -diff
        pivot = 1.0 + 2.0*diff + loss[j] - lo*c[j-1]
        c[j] = up/pivot; y[j] = (rhs[j] - lo*y[j-1])/pivot
    for j in range(size-2, -1, -1): y[j] -= c[j]*y[j+1]
    return y

@njit(cache=True)
def _state(u, initial, dt, h, a, positive):
    m,r,alpha,beta,q,mu1,mu2,gamma1,gamma2,lam,eta = a
    N,J = u.shape
    v = np.empty((2,N+1,J)); v[:,0,:] = initial
    zero = np.zeros(J); loss1 = np.full(J,dt*m)
    d1,d2 = dt*mu1/h**2,dt*mu2/h**2
    for n in range(N):
        if positive:
            v[1,n+1] = _solve(d2,dt*(beta*v[0,n]+q*u[n]),(1+dt*r)*v[1,n])
            v[0,n+1] = _solve(d1,loss1,v[0,n]+dt*alpha*v[0,n]*v[1,n+1])
        else:
            v[0,n+1] = _solve(d1,zero,v[0,n]+dt*v[0,n]*(-m+alpha*v[1,n]))
            v[1,n+1] = _solve(d2,zero,v[1,n]+dt*v[1,n]*(r-beta*v[0,n]-q*u[n]))
    return v

@njit(cache=True)
def _adjoint(u, v, targets, dt, h, a, positive, zeta):
    m,r,alpha,beta,q,mu1,mu2,gamma1,gamma2,lam,eta = a
    N,J = u.shape
    p = np.zeros_like(v); zero = np.zeros(J); loss1 = np.full(J,dt*m)
    d1,d2 = dt*mu1/h**2,dt*mu2/h**2
    if positive and zeta:
        p[1,N] = _solve(d2,dt*(beta*v[0,N-1]+q*u[N-1]),-dt*eta*q*u[N-1])
    for k in range(N-1,0,-1):
        if positive:
            rhs1 = p[0,k+1]+dt*(gamma1*(v[0,k]-targets[0,k])+v[1,k+1]*(alpha*p[0,k+1]-beta*p[1,k+1]))
            p[0,k] = _solve(d1,loss1,rhs1)
            rhs2 = (1+dt*r)*p[1,k+1]+dt*(gamma2*(v[1,k]-targets[1,k])-eta*q*((1-zeta)*u[k]+zeta*u[k-1])+alpha*v[0,k-1]*p[0,k])
            p[1,k] = _solve(d2,dt*(beta*v[0,k-1]+q*u[k-1]),rhs2)
        else:
            rhs1 = p[0,k+1]+dt*(gamma1*(v[0,k]-targets[0,k])+(-m+alpha*v[1,k])*p[0,k+1]-beta*v[1,k]*p[1,k+1])
            rhs2 = p[1,k+1]+dt*(gamma2*(v[1,k]-targets[1,k])-eta*q*u[k]+alpha*v[0,k]*p[0,k+1]+(r-beta*v[0,k]-q*u[k])*p[1,k+1])
            p[0,k] = _solve(d1,zero,rhs1); p[1,k] = _solve(d2,zero,rhs2)
    return p  # p[:,0] is unused because the initial state is fixed.

@njit(cache=True)
def _tangent(u, v, direction, dt, h, a, positive):
    m,r,alpha,beta,q,mu1,mu2,gamma1,gamma2,lam,eta = a
    N,J = u.shape
    z = np.zeros_like(v); zero = np.zeros(J); loss1 = np.full(J,dt*m)
    d1,d2 = dt*mu1/h**2,dt*mu2/h**2
    for n in range(N):
        if positive:
            rhs2 = (1+dt*r)*z[1,n]-dt*(beta*z[0,n]+q*direction[n])*v[1,n+1]
            z[1,n+1] = _solve(d2,dt*(beta*v[0,n]+q*u[n]),rhs2)
            rhs1 = z[0,n]+dt*alpha*(z[0,n]*v[1,n+1]+v[0,n]*z[1,n+1])
            z[0,n+1] = _solve(d1,loss1,rhs1)
        else:
            rhs1 = z[0,n]+dt*((-m+alpha*v[1,n])*z[0,n]+alpha*v[0,n]*z[1,n])
            rhs2 = z[1,n]+dt*(-beta*v[1,n]*z[0,n]+(r-beta*v[0,n]-q*u[n])*z[1,n]-q*v[1,n]*direction[n])
            z[0,n+1] = _solve(d1,zero,rhs1); z[1,n+1] = _solve(d2,zero,rhs2)
    return z

@njit(cache=True)
def _tangent_adjoint(u,v,p,z,direction,dt,h,a,positive,zeta):
    m,r,alpha,beta,q,mu1,mu2,gamma1,gamma2,lam,eta = a
    N,J = u.shape
    xi = np.zeros_like(v); zero = np.zeros(J); loss1 = np.full(J,dt*m)
    d1,d2 = dt*mu1/h**2,dt*mu2/h**2
    if positive and zeta:
        xi[1,N]=_solve(d2,dt*(beta*v[0,N-1]+q*u[N-1]),-dt*eta*q*direction[N-1]-dt*(beta*z[0,N-1]+q*direction[N-1])*p[1,N])
    for k in range(N-1,0,-1):
        if positive:
            rhs1=xi[0,k+1]+dt*(gamma1*z[0,k]+z[1,k+1]*(alpha*p[0,k+1]-beta*p[1,k+1])+v[1,k+1]*(alpha*xi[0,k+1]-beta*xi[1,k+1]))
            xi[0,k]=_solve(d1,loss1,rhs1)
            rhs2=(1+dt*r)*xi[1,k+1]+dt*(gamma2*z[1,k]-eta*q*((1-zeta)*direction[k]+zeta*direction[k-1])+alpha*z[0,k-1]*p[0,k]+alpha*v[0,k-1]*xi[0,k]-(beta*z[0,k-1]+q*direction[k-1])*p[1,k])
            xi[1,k]=_solve(d2,dt*(beta*v[0,k-1]+q*u[k-1]),rhs2)
        else:
            rhs1=xi[0,k+1]+dt*(gamma1*z[0,k]+alpha*z[1,k]*p[0,k+1]+(-m+alpha*v[1,k])*xi[0,k+1]-beta*z[1,k]*p[1,k+1]-beta*v[1,k]*xi[1,k+1])
            rhs2=xi[1,k+1]+dt*(gamma2*z[1,k]-eta*q*direction[k]+alpha*z[0,k]*p[0,k+1]+alpha*v[0,k]*xi[0,k+1]+(-beta*z[0,k]-q*direction[k])*p[1,k+1]+(r-beta*v[0,k]-q*u[k])*xi[1,k+1])
            xi[0,k]=_solve(d1,zero,rhs1); xi[1,k]=_solve(d2,zero,rhs2)
    return xi

@dataclass(frozen=True)
class Parameters:
    T: float=4.0
    m: float=1.0
    r: float=1.2
    alpha: float=1.0
    beta: float=1.0
    q: float=1.0
    mu1: float=0.002
    mu2: float=0.002
    gamma1: float=0.5
    gamma2: float=0.5
    lam: float=0.05
    eta: float=0.8
    umax: float=2.0
    def __post_init__(self):
        if min(self.T,self.m,self.r,self.alpha,self.beta,self.q,self.mu1,self.mu2,self.lam,self.umax)<=0:
            raise ValueError('Time, rates, diffusion, lambda, and the upper bound must be positive.')
        if min(self.gamma1,self.gamma2,self.eta)<0:
            raise ValueError('Tracking weights and harvest value must be nonnegative.')
    def vector(self)->Array:
        return np.array([self.m,self.r,self.alpha,self.beta,self.q,self.mu1,self.mu2,self.gamma1,self.gamma2,self.lam,self.eta])

class Problem:
    def __init__(self,M:int=80,N:int=120,scheme:Scheme='imex',parameters:Parameters|None=None,harvest_rule:Literal['left','flux']='left'):
        if M<2 or N<1 or scheme not in ('imex','positive'): raise ValueError('Require M >= 2, N >= 1 and a valid scheme.')
        if harvest_rule not in ('left','flux') or (scheme=='imex' and harvest_rule=='flux'):
            raise ValueError('Flux-aligned harvesting is implemented for the positive scheme only.')
        self.zeta=int(harvest_rule=='flux'); self.harvest_rule=harvest_rule
        self.M,self.N,self.scheme=M,N,scheme
        self.par=parameters or Parameters(); self.dt,self.h=self.par.T/N,1.0/M
        self.a=self.par.vector(); self.positive=scheme=='positive'
        self.x=np.linspace(0,1,M+1); self.t=np.arange(N)*self.dt
        self.w=np.full(M+1,self.h); self.w[[0,-1]]*=.5
        self.W=np.broadcast_to(self.dt*self.w,(N,M+1)).copy(); self.shape=(N,M+1)
        x=self.x
        self.initial=np.array([1.3*np.exp(-(x-.30)**2/(2*.10**2))+.05,1.4*np.exp(-(x-.70)**2/(2*.10**2))+.05])
        targets=np.array([self.par.r/self.par.beta+.15*np.cos(2*np.pi*x),self.par.m/self.par.alpha-.15*np.cos(2*np.pi*x)])
        self.targets=np.broadcast_to(targets[:,None,:],(2,N,M+1)).copy()
        self.forward_count=self.adjoint_count=self.hessian_count=0
    def control(self,u)->Array:
        a=np.asarray(u,dtype=float)
        if a.size!=self.N*(self.M+1) or not np.all(np.isfinite(a)): raise ValueError('Invalid control size or non-finite entries.')
        return np.ascontiguousarray(a.reshape(self.shape))
    def state(self,u)->Array:
        u=self.control(u); self.forward_count+=1
        v=_state(u,self.initial,self.dt,self.h,self.a,self.positive)
        if not np.all(np.isfinite(v)): raise FloatingPointError('Non-finite state; an IMEX trial may be unstable.')
        return v
    def adjoint(self,u,v)->Array:
        self.adjoint_count+=1
        return _adjoint(self.control(u),v,self.targets,self.dt,self.h,self.a,self.positive,self.zeta)
    def components(self,u,v)->Array:
        u=self.control(u); p=self.par
        return np.array([.5*p.gamma1*np.sum(self.W*(v[0,:-1]-self.targets[0])**2),.5*p.gamma2*np.sum(self.W*(v[1,:-1]-self.targets[1])**2),.5*p.lam*np.sum(self.W*u**2),-p.eta*p.q*np.sum(self.W*u*((1-self.zeta)*v[1,:-1]+self.zeta*v[1,1:]))])
    def objective(self,u,v=None)->float:
        if v is None: v=self.state(u)
        return float(self.components(u,v).sum())
    def gradient(self,u,v=None,p=None)->Array:
        u=self.control(u)
        if v is None: v=self.state(u)
        if p is None: p=self.adjoint(u,v)
        prey=v[1,1:] if self.positive else v[1,:-1]
        return self.par.lam*u-self.par.eta*self.par.q*((1-self.zeta)*v[1,:-1]+self.zeta*v[1,1:])-self.par.q*prey*p[1,1:]
    def evaluate(self,u):
        u=self.control(u); v=self.state(u); p=self.adjoint(u,v)
        return self.objective(u,v),self.gradient(u,v,p),v,p
    def hessian(self,u,direction,v=None,p=None)->Array:
        u=self.control(u); d=self.control(direction)
        if v is None: v=self.state(u)
        if p is None: p=self.adjoint(u,v)
        self.hessian_count+=1
        z=_tangent(u,v,d,self.dt,self.h,self.a,self.positive)
        xi=_tangent_adjoint(u,v,p,z,d,self.dt,self.h,self.a,self.positive,self.zeta)
        sl=slice(1,None) if self.positive else slice(None,-1)
        return self.par.lam*d-self.par.eta*self.par.q*((1-self.zeta)*z[1,:-1]+self.zeta*z[1,1:])-self.par.q*(z[1,sl]*p[1,1:]+v[1,sl]*xi[1,1:])
    def inner(self,u,v)->float: return float(np.sum(self.W*self.control(u)*self.control(v)))
    def norm(self,u)->float: return float(np.sqrt(max(0.0,self.inner(u,u))))
    def residual(self,u,g)->float:
        u=self.control(u)
        return float(np.max(np.abs(u-np.clip(u-g,0,self.par.umax))))
    def diagnostics(self,u,v,p,g)->dict:
        a=self.par; u=self.control(u)
        mass=a.beta*(v[0]@self.w)+a.alpha*(v[1]@self.w)
        sl=slice(1,None) if self.positive else slice(None,-1)
        rhs=-a.beta*a.m*(v[0,sl]@self.w)+a.alpha*a.r*(v[1,:-1]@self.w)-a.alpha*a.q*((u*v[1,sl])@self.w)
        return dict(objective=self.objective(u,v),residual=self.residual(u,g),min_state=float(v.min()),balance_error=float(np.max(np.abs(np.diff(mass)/self.dt-rhs))),chi1=float(np.min(1+self.dt*(-a.m+a.alpha*v[1,:-1]))),chi2=float(np.min(1+self.dt*(a.r-a.beta*v[0,:-1]-a.q*u))),components=self.components(u,v).tolist(),harvest=float(a.q*np.sum(self.W*u*((1-self.zeta)*v[1,:-1]+self.zeta*v[1,1:]))),harvest_rule=self.harvest_rule)

def optimize(problem:Problem,start=None,tol:float=1e-7,maxiter:int=1500,maxcor:int=38):
    """L-BFGS-B uses W*g; convergence is checked again using the weighted residual."""
    if start is None: start=np.zeros(problem.shape)
    u0=problem.control(start); history=[]; cache={}
    before=(problem.forward_count,problem.adjoint_count); clock=perf_counter()
    def fun(x):
        f,g,v,p=problem.evaluate(x); cache.update(x=x.copy(),f=f,g=g,v=v,p=p)
        return f,(problem.W*g).ravel()
    def callback(x):
        if not np.array_equal(cache.get('x'),x): fun(x)
        history.append(problem.residual(x,cache['g']))
    result=minimize(fun,u0.ravel(),jac=True,method='L-BFGS-B',bounds=[(0,problem.par.umax)]*u0.size,callback=callback,
                    options=dict(maxcor=maxcor,ftol=0.0,gtol=tol*float(problem.W.min()),maxiter=maxiter,maxls=60))
    u=result.x.reshape(problem.shape); f,g,v,p=problem.evaluate(u)
    report=problem.diagnostics(u,v,p,g)
    report.update(iterations=int(result.nit),forward=problem.forward_count-before[0],adjoint=problem.adjoint_count-before[1],seconds=perf_counter()-clock,message=str(result.message),history=history,accepted=bool(problem.residual(u,g)<=1e-6))
    return u,report

def sweep(problem:Problem,theta:float,start=None,tol=1e-6,maxiter=12000,anderson=False,depth=10,omega=.8,regularization=1e-10):
    """Relaxation is inside the projection; Anderson is unsafeguarded type II."""
    u=np.zeros(problem.shape) if start is None else problem.control(start).copy()
    past_u=[]; past_f=[]; history=[]
    before=(problem.forward_count,problem.adjoint_count); clock=perf_counter()
    for k in range(maxiter+1):
        f,g,v,p=problem.evaluate(u); residual=problem.residual(u,g); history.append(residual)
        if residual<=tol or k==maxiter: break
        G=np.clip(u-theta/problem.par.lam*g,0,problem.par.umax); fk=(G-u).ravel()
        if anderson and past_f:
            us=past_u+[u.ravel().copy()]; fs=past_f+[fk.copy()]
            DU=np.column_stack([us[j+1]-us[j] for j in range(len(us)-1)])
            DF=np.column_stack([fs[j+1]-fs[j] for j in range(len(fs)-1)])
            coeff=np.linalg.solve(DF.T@DF+regularization*np.eye(DF.shape[1]),DF.T@fk)
            new=np.clip((G.ravel()-omega*(DU+DF)@coeff).reshape(problem.shape),0,problem.par.umax)
        else: new=G
        past_u.append(u.ravel().copy()); past_f.append(fk.copy())
        past_u=past_u[-depth:]; past_f=past_f[-depth:]; u=new
    report=problem.diagnostics(u,v,p,g)
    report.update(iterations=k,forward=problem.forward_count-before[0],adjoint=problem.adjoint_count-before[1],seconds=perf_counter()-clock,history=history,theta=theta,accepted=bool(residual<=tol))
    return u,report

def spg(problem:Problem,start=None,tol=1e-6,maxiter=5000):
    u=np.zeros(problem.shape) if start is None else problem.control(start).copy()
    before=(problem.forward_count,problem.adjoint_count); clock=perf_counter()
    f,g,v,p=problem.evaluate(u); history=[]; values=[f]; step=.5
    for k in range(maxiter+1):
        residual=problem.residual(u,g); history.append(residual)
        if residual<=tol or k==maxiter: break
        d=np.clip(u-step*g,0,problem.par.umax)-u; gd=problem.inner(g,d); sigma=1.0
        for trial in range(60):
            new=u+sigma*d; vn=problem.state(new); fn=problem.objective(new,vn)
            if fn<=max(values[-20:])+1e-4*sigma*gd: break
            sigma*=.5
        else: raise RuntimeError('SPG line search failed.')
        pn=problem.adjoint(new,vn); gn=problem.gradient(new,vn,pn)
        s=new-u; y=gn-g; sy=problem.inner(s,y)
        if sy>0:
            step=problem.inner(s,s)/sy if k%2==0 else sy/problem.inner(y,y)
            step=float(np.clip(step,1e-8,1e4))
        else: step=.5
        u,f,g,v,p=new,fn,gn,vn,pn; values.append(f)
    report=problem.diagnostics(u,v,p,g)
    report.update(iterations=k,forward=problem.forward_count-before[0],adjoint=problem.adjoint_count-before[1],seconds=perf_counter()-clock,history=history,accepted=bool(residual<=tol))
    return u,report

def free_spectrum(problem:Problem,u,tol=1e-8):
    u=problem.control(u); f,g,v,p=problem.evaluate(u)
    free=(u>1e-8)&(u<problem.par.umax-1e-8)
    idx=np.flatnonzero(free.ravel()); weights=problem.W.ravel()[idx]; root=np.sqrt(weights)
    if idx.size<2: raise ValueError('Fewer than two free coordinates.')
    def mv(x):
        d=np.zeros(u.size); d[idx]=x/root
        return root*problem.hessian(u,d,v,p).ravel()[idx]
    op=LinearOperator((idx.size,idx.size),matvec=mv,dtype=np.float64)
    v0=np.random.default_rng(20260914).normal(size=idx.size)
    lo,lv=eigsh(op,k=1,which='SA',tol=tol,maxiter=4000,v0=v0)
    hi,hv=eigsh(op,k=1,which='LA',tol=tol,maxiter=2000,v0=v0)
    report=dict(mu_min=float(lo[0]),mu_max=float(hi[0]),free=int(idx.size),lower=int(np.sum(u<=1e-8)),upper=int(np.sum(u>=problem.par.umax-1e-8)),min_eigen_residual=float(np.linalg.norm(mv(lv[:,0])-lo[0]*lv[:,0])),max_eigen_residual=float(np.linalg.norm(mv(hv[:,0])-hi[0]*hv[:,0])))
    if lo[0]>0:
        theta_opt = float(min(1.0, 2*problem.par.lam/(lo[0]+hi[0])))
        rho_opt = float(max(abs(1-theta_opt*lo[0]/problem.par.lam),
                            abs(1-theta_opt*hi[0]/problem.par.lam)))
        report.update(theta_max=float(2*problem.par.lam/hi[0]),
                      theta_opt=theta_opt, rho_opt=rho_opt)
    lower=u<=1e-8; upper=u>=problem.par.umax-1e-8
    report['lower_gradient_margin']=float(g[lower].min()) if lower.any() else None
    report['upper_gradient_margin']=float((-g[upper]).min()) if upper.any() else None
    d=np.zeros(u.size); d[idx]=hv[:,0]/root
    return report,d.reshape(problem.shape)
