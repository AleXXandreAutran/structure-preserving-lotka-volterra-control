"""Reproduce article v2. All reported values are computed by this script.

Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiments.py
Use --phase reference|derivatives|grids|coarse|fixed|multistart to run a subset.
The optional --reuse flag preserves already computed reference solutions.
"""
from pathlib import Path
from time import perf_counter
import argparse,json,sys,platform
import numpy as np
import scipy,numba
from lotka_v2 import Problem,optimize,sweep,spg,free_spectrum
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
LABELS=['imex','positive','flux']
def problem(label,M=80,N=120):return Problem(M,N,'positive' if label=='flux' else label,harvest_rule='flux' if label=='flux' else 'left')
def save(name,data):(OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
def load(name,default=None):
    path=OUT/(name+'.json');return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)
def refcontrol(label):return np.load(OUT/f'reference_{label}.npz')['u']
def warm():
    for s in LABELS:
        P=problem(s,4,6);u=np.full(P.shape,.7);f,g,v,p=P.evaluate(u);P.hessian(u,np.ones(P.shape),v,p)

def reference(reuse=False):
    rr=load('reference') if reuse else {};spec=load('spectra') if reuse else {};algs=load('algorithms') if reuse else {}
    for s in LABELS:
        P=problem(s)
        if s in rr and reuse:u=refcontrol(s);r=rr[s]
        else:
            u,r=optimize(P);rr[s]=r;np.savez_compressed(OUT/f'reference_{s}.npz',u=u,v=P.state(u));save('reference',rr)
        print('REFERENCE',s,{k:r[k] for k in ['objective','residual','iterations','min_state']},flush=True)
        if s in spec and reuse:d=np.load(OUT/f'spectrum_{s}.npz')['direction']
        else:
            spec[s],d=free_spectrum(P,u,tol=1e-6)
            np.savez_compressed(OUT/f'spectrum_{s}.npz',direction=d);save('spectra',spec)
        print('SPECTRUM',s,spec[s],flush=True)
        algs[s]={'L-BFGS-B':r}
        _,algs[s]['FBS']=sweep(P,.0093,maxiter=12000)
        _,algs[s]['Anderson-FBS']=sweep(P,.01,anderson=True,maxiter=12000)
        _,algs[s]['SPG']=spg(P,maxiter=10000)
        delta=1e-5/max(1,np.max(np.abs(d)));local=[]
        for ratio in [.95,1.05]:
            theta=ratio*spec[s]['theta_max']
            def G(x):return np.clip(x-theta/P.par.lam*P.gradient(x),0,P.par.umax)
            measured=P.norm((G(u+delta*d)-G(u))/delta)/P.norm(d)
            local.append(dict(theta=theta,ratio=ratio,predicted=abs(1-theta*spec[s]['mu_max']/P.par.lam),measured=measured))
        spec[s]['local_threshold_test']=local;save('spectra',spec);save('algorithms',algs)
        V=problem('imex',80,1920);ur=np.repeat(u,16,axis=0);vr=V.state(ur)
        rr[s]['validation_objective']=V.objective(ur,vr)
        rr[s]['validation_min_state']=float(vr.min())
        rr[s]['temporal_variation']=float(np.sum(P.w*np.abs(np.diff(u,axis=0))))
        save('reference',rr)
    P=problem('imex');save('cross_reference',{s:P.norm(refcontrol('imex')-refcontrol(s)) for s in LABELS[1:]})

def derivatives():
    tests={};rng=np.random.default_rng(20260914);P=problem('imex');xx,tt=np.meshgrid(P.x,P.t)
    d=.35*np.cos(2*np.pi*xx)*np.cos(np.pi*tt/P.par.T)+.1*rng.normal(size=P.shape)
    d/=np.max(np.abs(d));e=rng.normal(size=P.shape);e/=np.max(np.abs(e))
    for s in LABELS:
        P=problem(s);u=np.full(P.shape,.7);f,g,v,p=P.evaluate(u);Hd=P.hessian(u,d,v,p);He=P.hessian(u,e,v,p);rows=[]
        for ep in np.logspace(-1,-7,13):
            fe,ge,_,_=P.evaluate(u+ep*d)
            rows.append(dict(epsilon=float(ep),first_remainder=abs(fe-f-ep*P.inner(g,d)),second_remainder=abs(fe-f-ep*P.inner(g,d)-.5*ep**2*P.inner(Hd,d)),gradient_remainder=P.norm(ge-g-ep*Hd)))
        slope=lambda key:float(np.polyfit(np.log([x['epsilon'] for x in rows[:5]]),np.log([x[key] for x in rows[:5]]),1)[0])
        sym=abs(P.inner(Hd,e)-P.inner(d,He))/max(1,abs(P.inner(Hd,e)),abs(P.inner(d,He)))
        tests[s]=dict(rows=rows,first_slope=slope('first_remainder'),second_slope=slope('second_remainder'),gradient_slope=slope('gradient_remainder'),weighted_symmetry_defect=sym)
    save('derivative_tests',tests);print('DERIVATIVES',{s:{k:v for k,v in x.items() if k!='rows'} for s,x in tests.items()},flush=True)

def grids():
    rr=load('reference');grid=[];previous={}
    for N in [30,60,120,240,480,960]:
        row=dict(N=N,dt=4/N);uu={}
        for s in LABELS:
            P=problem(s,80,N)
            if N==120:u=refcontrol(s);r=dict(rr[s])
            else:
                start=np.repeat(previous[s],2,axis=0) if s in previous else None
                u,r=optimize(P,start,maxiter=3000)
            uu[s]=u;previous[s]=u
            np.savez_compressed(OUT/f'grid_{s}_M80_N{N}.npz',u=u)
            V=problem('imex',80,1920);ur=np.repeat(u,1920//N,axis=0)
            r['validation_objective']=V.objective(ur);row[s]=r
            print('GRID RUN',N,s,r['objective'],r['residual'],r['iterations'],flush=True)
        P=problem('imex',80,N)
        for s in LABELS[1:]:
            row['control_distance_'+s]=P.norm(uu['imex']-uu[s]);row['objective_gap_'+s]=abs(row['imex']['objective']-row[s]['objective'])
        grid.append(row);save('time_grid',grid)

def coarse():
    rows=[]
    for N in [4,6,8,10,20]:
        row=dict(N=N,dt=4/N)
        for s in LABELS:
            P=problem(s,80,N);u,r=optimize(P,maxiter=3000)
            V=problem('imex',80,1920);uf=np.repeat(u,1920//N,axis=0);vf=V.state(uf)
            r.update(validation_objective=V.objective(uf,vf),validation_min_state=float(vf.min()));row[s]=r
            np.savez_compressed(OUT/f'coarse_{s}_N{N}.npz',u=u,v=P.state(u))
            print('COARSE',N,s,r['objective'],r['min_state'],r['residual'],flush=True)
        rows.append(row);save('coarse_optimization',rows)
    positivity=[]
    for N in [4,6,8,10,12,16,20,30,60,120]:
        row=dict(N=N,dt=4/N)
        for s in LABELS[:2]:
            P=problem(s,80,N);u=np.full(P.shape,2.);f,g,v,p=P.evaluate(u);row[s]=P.diagnostics(u,v,p,g)
        positivity.append(row)
    save('positivity',positivity)

def fixed():
    rows=[]
    def state(M,N,s):
        P=problem(s,M,N);x,t=np.meshgrid(P.x,P.t)
        u=.7+.25*np.sin(np.pi*x)*np.cos(2*np.pi*t/P.par.T)
        return P,P.state(u)
    for s in LABELS[:2]:
        for N in [60,120,240,480]:
            P,v=state(160,N,s);_,vf=state(160,N*2,s)
            rows.append(dict(scheme=s,study='time',grid=N,error=float(np.sqrt(np.sum((v[:,-1]-vf[:,-1])**2*P.w)))))
        for M in [20,40,80,160]:
            P,v=state(M,960,s);_,vf=state(2*M,960,s)
            rows.append(dict(scheme=s,study='space',grid=M,error=float(np.sqrt(np.sum((v[:,-1]-vf[:,-1,::2])**2*P.w)))))
    save('fixed_control',rows)

def multistart():
    rows={};rr=load('reference')
    for s in LABELS:
        P=problem(s);rng=np.random.default_rng(20260806)
        starts={'zero':np.zeros(P.shape),'one':np.ones(P.shape),'upper':np.full(P.shape,2.),'uniform_random':rng.uniform(0,2,P.shape),'front_loaded':np.broadcast_to(np.where(P.t<P.par.T/2,2.,0.)[:,None],P.shape)}
        rows[s]=[]
        for name,start in starts.items():
            if name=='zero':u=refcontrol(s);r=dict(rr[s])
            else:u,r=optimize(P,start,maxiter=3000)
            r.update(start=name,distance_to_zero_start=P.norm(u-refcontrol(s)));rows[s].append(r)
            print('MULTISTART',s,name,r['objective'],r['residual'],r['iterations'],flush=True)
        save('multistart',rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','reference','derivatives','grids','coarse','fixed','multistart'],default='all');parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
    warm();metadata=load('environment');metadata.update(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,platform=platform.platform(),seed=20260914,implementation='standalone manuscript-equation implementation');clock=perf_counter()
    jobs=dict(reference=lambda:reference(args.reuse),derivatives=derivatives,grids=grids,coarse=coarse,fixed=fixed,multistart=multistart)
    for key,job in jobs.items():
        if args.phase in ('all',key):job()
    metadata['last_phase']=args.phase;metadata['last_phase_seconds']=perf_counter()-clock;save('environment',metadata)
if __name__=='__main__':main()
