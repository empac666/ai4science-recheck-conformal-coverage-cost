from common import *

def threshold(scores,legacy=False):
    assert scores.ndim==1 and len(scores)>0
    assert not np.isnan(scores).any() and not np.isneginf(scores).any()
    n=len(scores); k=(9*(n+1)+9)//10
    if legacy:
        # legacy higher selects one order statistic, including explicit +infinity.
        Q=float(np.quantile(scores,min(1,k/n),method='higher'))
    else:
        Q=math.inf if np.isfinite(scores).sum()<k else float(np.sort(scores)[k-1])
    assert math.isfinite(Q) or Q==math.inf
    return Q,k

def evaluate(ql,qh,y,L,U,Q,method):
    assert all(np.isfinite(x).all() for x in (ql,qh,y,L,U)) and np.all(L<=U)
    n=len(y); fallback=np.full(n,Q==math.inf and method!='U',dtype=bool)
    if Q==math.inf:
        prelo=np.full(n,-math.inf); prehi=np.full(n,math.inf)
        lo=L.copy() if method!='U' else prelo.copy()
        hi=U.copy() if method!='U' else prehi.copy()
        preempty=np.zeros(n,dtype=bool); outside=np.zeros(n,dtype=bool)
    else:
        prelo=ql-Q; prehi=qh+Q
        preempty=prelo>prehi; outside=(prelo>U)|(prehi<L)
        lo=np.maximum(prelo,L) if method!='U' else prelo.copy()
        hi=np.minimum(prehi,U) if method!='U' else prehi.copy()
    empty=lo>hi
    if method=='U' and Q==math.inf:
        delta=np.full(n,math.inf); width=delta.copy(); score=delta.copy(); wset=delta.copy()
    else:
        delta=np.subtract(hi,lo,dtype=np.float64) # Single endpoint subtraction.
        width=np.where(empty,math.inf,delta)
        score=np.where(empty,math.inf,delta+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0))
        wset=np.maximum(delta,0)
    near=empty&(delta>=-EPS)&(delta<0)
    hit=(lo<=y)&(y<=hi)
    full=(~fallback)&(~empty)&(lo==L)&(hi==U)
    return dict(q_lo=ql,q_hi=qh,y=y,L=L,U=U,lo=lo,hi=hi,endpoint_delta=delta,empty=empty,near_zero_empty=near,hit=hit,w=width,W_set_row=wset,IS=score,corridor_width=U-L,base_cross=ql>qh,preclip_empty=preempty,outside_corridor=outside,fallback=fallback,full_corridor=full,clip_active=(lo!=prelo)|(hi!=prehi),endpoint_violation=(lo<L)|(hi>U),outside_y=(y<L)|(y>U))

def summarize(a,Q,k,eta_cal,cal_n,cal_group_n):
    out={'C':float(a['hit'].mean()),'W_raw':float(a['endpoint_delta'].mean()),'W_set':float(a['W_set_row'].mean()),'W':float(a['w'].mean()),'IS':float(a['IS'].mean()),'E':float(a['empty'].mean()),'corridor_mean_width':float(a['corridor_width'].mean()),'eta_test':float(a['outside_y'].mean()),'eta_cal':float(eta_cal),'Q':Q,'k':k,'cal_n':cal_n,'cal_group_n':cal_group_n,'test_n':len(a['y'])}
    for col in ('empty','near_zero_empty','base_cross','preclip_empty','outside_corridor','fallback','full_corridor','clip_active','endpoint_violation'):
        out[col+'_rows']=int(a[col].sum()); out[col+'_rate']=float(a[col].mean())
    mask=~a['empty']; out['nonempty_IS']=float(a['IS'][mask].mean()) if mask.any() else {'value':None,'state':'missing/error','reason':'no_nonempty_rows'}
    out['F_cal']=int(a['fallback'].any()); out['F_row']=float(a['fallback'].mean())
    return out

def check_examples():
    for n,k in [(8,9),(9,9),(10,10),(19,18),(20,19)]: assert threshold(np.arange(n,dtype=float))[1]==k
    assert max(10-15,15-20)==-5
    assert threshold(np.arange(-1,9,dtype=float))[0]==8
    a=np.arange(-1,9,dtype=float); a[-1]=math.inf; assert threshold(a)[0]==math.inf
    assert threshold(np.arange(8,dtype=float))[0]==math.inf
    assert threshold(np.arange(20,dtype=float))[0]==18 and threshold(np.arange(20,dtype=float),True)[0]==19
    z=evaluate(np.array([10.]),np.array([20.]),np.array([25.]),np.array([0.]),np.array([30.]),0.,'N')
    assert z['IS'][0]==110
    for method in ('U','N','R'):
        z=evaluate(np.array([1.]),np.array([2.]),np.array([1.5]),np.array([0.]),np.array([3.]),math.inf,method)
        assert z['hit'][0] and not z['empty'][0]
    z=evaluate(np.array([2.]),np.array([1.]),np.array([1.5]),np.array([0.]),np.array([3.]),0.,'N')
    assert z['empty'][0] and z['w'][0]==math.inf and z['IS'][0]==math.inf
    z=evaluate(np.array([1.+5e-9]),np.array([1.]),np.array([1.]),np.array([0.]),np.array([3.]),0.,'N')
    assert z['near_zero_empty'][0] and not z['hit'][0]
    return True
