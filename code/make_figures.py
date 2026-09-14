"""Plot only results produced by run_experiments.py"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
LABEL={'imex':'IMEX','positive':r'Positive, left ($P_0$)','flux':r'Positive, flux ($P_1$)'}
def load(name):return json.loads((RES/(name+'.json')).read_text())
def new(size=(4.7,3.3)):return plt.figure(figsize=size,layout='constrained'),plt.gca()
def finish(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)
def main():
    tests=load('derivative_tests')
    for key,ylab,name in [('gradient_remainder',r'$\|g(u+\varepsilon d)-g(u)-\varepsilon Hd\|_{h,\tau}$','hessian_taylor'),('second_remainder',r'Second-order objective remainder $E_2(\varepsilon)$','scalar_taylor')]:
        fig,ax=new()
        for s in LABEL:
            rows=tests[s]['rows'];ax.loglog([r['epsilon'] for r in rows],[max(r[key],1e-19) for r in rows],'-o',markersize=3,label=LABEL[s])
        order=2 if key=='gradient_remainder' else 3
        ep=np.logspace(-5,-1,30);base=tests['imex']['rows'][0][key]/(.1**order)
        ax.loglog(ep,base*ep**order,'--',linewidth=1,label=f'Order {order}')
        ax.set(xlabel=r'Perturbation $\varepsilon$',ylabel=ylab);ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=8)
        finish(fig,name)
    data=load('positivity');fig,ax=new()
    for s in ['imex','positive']:
        ax.plot([r['dt'] for r in data],[r[s]['min_state'] for r in data],'-o',markersize=3,label=LABEL[s])
    ax.axhline(0,linestyle='--',linewidth=.8);ax.set(xlabel=r'Time step $\tau$',ylabel='Minimum state value');ax.legend(fontsize=8);ax.grid(alpha=.25);finish(fig,'positivity')
    spec=load('spectra');fig,ax=new();xx=np.linspace(.001,1.1,500)
    for s in LABEL:
        lo,hi=spec[s]['mu_min'],spec[s]['mu_max'];rho=np.maximum(np.abs(1-2*xx*lo/hi),np.abs(1-2*xx))
        ax.plot(xx,rho,label=LABEL[s])
    ax.axvline(1,linestyle='--',linewidth=.8);ax.set(xlabel=r'$\theta/\theta_{\max}$',ylabel='Predicted free linear factor',ylim=(.90,1.13));ax.grid(alpha=.25);ax.legend(fontsize=8);finish(fig,'spectral_factor')
    algs=load('algorithms')
    for s in LABEL:
        fig,ax=new((3.2,2.8))
        for name,r in algs[s].items():
            ax.semilogy(np.arange(len(r['history'])),np.maximum(r['history'],1e-12),label=name,linewidth=1.1)
        ax.axhline(1e-6,linestyle='--',linewidth=.8)
        ax.set(xlabel='Accepted iteration',ylabel=r'$\|R_{h,\tau}(u)\|_\infty$',title=LABEL[s]);ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=9)
        finish(fig,'algorithms_'+s)
    for s in LABEL:
        u=np.load(RES/f'reference_{s}.npz')['u'];N,J=u.shape
        x=np.linspace(0,1,J);edges=np.r_[0,(x[:-1]+x[1:])/2,1]
        fig,ax=new((3.2,2.8));m=ax.pcolormesh(np.linspace(0,4,N+1),edges,u.T,shading='flat',vmin=0,vmax=2,rasterized=True)
        cb=fig.colorbar(m,ax=ax);cb.set_label('Fishing effort')
        ax.set(xlabel='Time',ylabel='Space',title=LABEL[s]);finish(fig,'control_'+s)
    data=load('time_grid');fig,ax=new()
    for s in ['positive','flux']:
        ax.loglog([r['dt'] for r in data],[r['control_distance_'+s] for r in data],'-o',markersize=4,label=r'$\|u^{\rm I}-u^{'+('P_0' if s=='positive' else 'P_1')+r'}\|_{h,\tau}$')
    bad=[r for r in data if not r['positive']['accepted']]
    if bad:ax.loglog([r['dt'] for r in bad],[r['control_distance_positive'] for r in bad],'x',markersize=9,label=r'$P_0$: residual above tolerance')
    ep=np.array([data[-1]['dt'],data[0]['dt']]);ax.loglog(ep,2.15*ep,'--',linewidth=1,label='Order 1')
    ax.set(xlabel=r'Time step $\tau$',ylabel='Control difference on the same grid');ax.legend(fontsize=7.5);ax.grid(True,which='both',alpha=.25);finish(fig,'control_refinement')
    fig,ax=new()
    for s in LABEL:ax.plot([r['N'] for r in data],[r[s]['validation_objective'] for r in data],'-o',markersize=4,label=LABEL[s])
    ax.set(xlabel='Optimization time steps N',ylabel='Common fine-grid objective',xscale='log');ax.legend(fontsize=8);ax.grid(alpha=.25);finish(fig,'validation_refinement')
    fixed=load('fixed_control')
    for study,exponent in [('time',1),('space',2)]:
        fig,ax=new()
        for s in ['imex','positive']:
            rows=[r for r in fixed if r['study']==study and r['scheme']==s]
            mesh=[4/r['grid'] if study=='time' else 1/r['grid'] for r in rows]
            ax.loglog(mesh,[r['error'] for r in rows],'-o',markersize=4,label=LABEL[s])
        ax.set(xlabel=r'$\tau$' if study=='time' else r'$h$',ylabel='Fixed-control final-state difference');ax.legend(fontsize=8);ax.grid(True,which='both',alpha=.25);finish(fig,'state_'+study)
    print(f'{len(list(FIG.glob("*.pdf")))} figure PDFs and matching PNGs saved in {FIG}.')
if __name__=='__main__':main()
