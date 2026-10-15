"""Independent reader of newly generated lossless row tables (O1--O7)."""
from common import *
from judge import judge, UNITS
import math

def equal(a,b,tol=EPS):
    a=np.asarray(a); b=np.asarray(b)
    assert a.shape==b.shape
    assert np.array_equal(np.isposinf(a),np.isposinf(b))
    assert not np.isnan(a).any() and not np.isnan(b).any()
    mask=np.isfinite(a)&np.isfinite(b)
    assert np.all(np.abs(a[mask]-b[mask])<=tol),'O6: independent mismatch'

def recompute(task,meta,arrays,pred,cal_ids):
    """No production metric or interval functions are called here."""
    z=arrays; y=z['y']; l=z['lo']; h=z['hi']; d=z['endpoint_delta']
    e=l>h; near=e&(d>=-1e-8)&(d<0)
    assert np.array_equal(z['empty'],e) and np.array_equal(z['near_zero_empty'],near)
    assert np.array_equal(z['hit'],(l<=y)&(y<=h))
    qlo,qhi=z['q_lo'],z['q_hi']; L,U=z['L'],z['U']
    assert np.array_equal(z['original_id'],pred['test_ids']) and np.array_equal(z['anonymous_group'],pred['test_group']),'O2 row IDs'
    assert np.array_equal(y,pred['test_y']) and np.array_equal(qlo,pred['test_lo']) and np.array_equal(qhi,pred['test_hi'])
    assert np.isfinite(qlo).all() and np.isfinite(qhi).all()
    calpos={int(v):i for i,v in enumerate(pred['cal_ids'])}; ix=np.array([calpos[int(v)] for v in cal_ids])
    yc=pred['cal_y'][ix]; ca=pred['cal_lo'][ix]; cb=pred['cal_hi'][ix]
    raw=np.maximum(ca-yc,yc-cb); assert np.isfinite(raw).all()
    if task['branch']=='FRP':
        xc=pred['cal_X'][ix]; vf,ef,em=xc[:,0],xc[:,1],xc[:,2]
        lc=1/(vf/ef+(1-vf)/em); vmax=vf*ef+(1-vf)*em; uc=lc+task['rho']*(vmax-lc)
    else: lc=np.zeros(len(ix)); uc=np.full(len(ix),meta['upper_bound'])
    outside=(yc<lc)|(yc>uc)
    k=(9*(len(ix)+1)+9)//10
    def rank(v):
        if task['legacy']: return float(np.quantile(v,min(1,k/len(v)),method='higher'))
        return math.inf if np.isfinite(v).sum()<k else float(sorted(v)[k-1])
    QN=rank(raw); adj=raw.copy(); adj[outside]=math.inf; QR=rank(adj)
    assert QR>=QN-EPS if math.isfinite(QN) else QR==math.inf,'O4 threshold'
    Q=QR if task['method']=='R' else QN
    equal(Q,meta['Q']); assert k==meta['k']
    if Q==math.inf:
        el=L if task['method']!='U' else np.full(len(y),-math.inf)
        eh=U if task['method']!='U' else np.full(len(y),math.inf)
    else:
        el=qlo-Q; eh=qhi+Q
        if task['method']!='U': el=np.maximum(el,L); eh=np.minimum(eh,U)
    expected_flags={'base_cross':qlo>qhi,'preclip_empty':np.zeros(len(y),dtype=bool) if Q==math.inf else (qlo-Q>qhi+Q),'outside_corridor':np.zeros(len(y),dtype=bool) if Q==math.inf else ((qlo-Q>U)|(qhi+Q<L)),'fallback':np.full(len(y),Q==math.inf and task['method']!='U',dtype=bool),'outside_y':(y<L)|(y>U),'endpoint_violation':(l<L)|(h>U)}
    expected_flags['full_corridor']=(~expected_flags['fallback'])&(~e)&(l==L)&(h==U)
    expected_flags['clip_active']=np.ones(len(y),dtype=bool) if Q==math.inf and task['method']!='U' else np.zeros(len(y),dtype=bool) if Q==math.inf else (l!=qlo-Q)|(h!=qhi+Q)
    for name,values in expected_flags.items(): assert np.array_equal(z[name],values),'O6 root flag '+name
    equal(z['corridor_width'],U-L,tol=0)
    # Negative infinity is legal only for the explicitly unbounded U branch.
    assert np.array_equal(l,el) and np.array_equal(h,eh),'O5: endpoints'
    unbounded=task['method']=='U' and Q==math.inf
    if unbounded:
        assert np.isneginf(l).all() and np.isposinf(h).all()
        wd=np.full(len(y),math.inf); isc=wd.copy()
        equal(d,wd)
    else:
        assert np.isfinite(l).all() and np.isfinite(h).all()
        equal(d,np.subtract(h,l,dtype=np.float64),tol=0)
        wd=np.where(e,math.inf,d)
        isc=d.copy(); below=y<l; above=y>h
        isc[below]+=20*(l[below]-y[below]); isc[above]+=20*(y[above]-h[above]); isc[e]=math.inf
    equal(z['w'],wd); equal(z['IS'],isc)
    equal(z['W_set_row'],np.maximum(d,0))
    if task['method']!='U':
        assert np.all(l[~e]>=L[~e]-EPS) and np.all(h[~e]<=U[~e]+EPS)
        if QR==math.inf: rl,rh=L,U
        else: rl,rh=np.maximum(qlo-QR,L),np.minimum(qhi+QR,U)
        if QN==math.inf: nl,nh=L,U
        else: nl,nh=np.maximum(qlo-QN,L),np.minimum(qhi+QN,U)
        ne=nl>nh
        assert not np.any((~ne)&(rl>rh))
        assert np.all(rl[~ne]<=nl[~ne]+EPS) and np.all(rh[~ne]>=nh[~ne]-EPS)
    result={'C':float(z['hit'].mean()),'W_raw':float(d.mean()),'W_set':float(np.maximum(d,0).mean()),'W':float(wd.mean()),'IS':float(isc.mean()),'E':float(e.mean()),'test_n':len(y),'empty_rows':int(e.sum()),'near_zero_empty_rows':int(near.sum()),'F_cal':int(z['fallback'].any()),'F_row':float(z['fallback'].mean()),'eta_cal':float(outside.mean()),'eta_test':float(((y<L)|(y>U)).mean()),'corridor_mean_width':float(z['corridor_width'].mean())}
    for field in ('base_cross','preclip_empty','outside_corridor','fallback','full_corridor','clip_active','endpoint_violation'):
        result[field+'_rows']=int(z[field].sum()); result[field+'_rate']=float(z[field].mean())
    for key,value in result.items(): equal(value,meta['metrics'][key],tol=1e-10 if key=='C' else EPS)
    return result

def val(x):
    if isinstance(x,dict):
        assert x=={'value':None,'state':'positive_infinity'}; return math.inf
    return x

def bootstrap(v,J):
    v=np.asarray(v,dtype=float)
    assert np.isfinite(v).all(),'bootstrap must short circuit infinity'
    means=v[J].mean(axis=1); assert np.isfinite(means).all()
    return [float(x) for x in np.percentile(means,[2.5,97.5],method='linear')]

def infinity_detail(by):
    result={}
    for m in ('N','R'):
        rows=[]; wi=[]; ii=[]
        for s in range(100,120):
            z=by[(s,m)]; rows.append({k:int(z[k]) for k in ('test_n','empty_rows','near_zero_empty_rows')}|{'seed':s})
            if math.isinf(z['W']): wi.append(s)
            if math.isinf(z['IS']): ii.append(s)
        result[m]={'rows':rows,'W_infinite_seed_ids':wi,'W_infinite_seed_count':len(wi),'IS_infinite_seed_ids':ii,'IS_infinite_seed_count':len(ii)}
    return result

def branch_primary(branch,tasks,independent):
    selected=[t for t in tasks if t['branch']==branch and t['mono'] and t['p']==93 and t['n']=='all' and t['namespace']=='formal']
    expected=18 if branch=='A' else 60
    assert len(selected)==expected and all(t['key'] in independent for t in selected)
    out={}
    for t in selected:
        key=t['key']; meta=read(ROOT/'results'/f'{key}.json')
        assert meta['rows_sha256']==filehash(ROOT/'rows'/f'{key}.npz'),'O7 cache hash'
        # Results entering the judge are derived from the separate row-reader.
        out[(t['seed'],t['method'])]=independent[key]
    return out

def object_lock(kind,tasks,independent):
    path=ROOT/'objects'/f'{kind}.json'; assert not path.exists(),'immutable object lock'
    by=branch_primary('A' if kind=='A' else 'B',tasks,independent)
    lock=read(ROOT/'locks/environment.json')
    env=versions()==lock['versions']; exact=versions()==PAPER
    r={'delivered':True,'object_audit_pass':True,'locked_environment_match':env,'exact_environment':exact,'units':UNITS[kind]}
    J=np.load(ROOT/'locks/bootstrap_J.npy',allow_pickle=False)
    seeds=list(range(6)) if kind=='A' else list(range(100,120))
    def vec(m,f,source=by): return np.array([source[(s,m)][f] for s in seeds])
    if kind=='A':
        cn,cr=vec('N','C').mean(),vec('R','C').mean(); wn,wr=vec('N','W_raw').mean(),vec('R','W_raw').mean()
        r['A_vector']=[float(100*cn),float(100*cr),float(100*(cr-cn)),float(wn),float(wr),float(wr-wn)]
    elif kind=='B':
        r['ci_R_cover']=bootstrap(100*vec('R','C'),J)
        r['ci_gain']=bootstrap(100*(vec('R','C')-vec('N','C')),J)
        w1,w2=vec('N','W'),vec('R','W'); inf=not(np.isfinite(w1).all() and np.isfinite(w2).all())
        r['any_N_R_width_infinite']=inf; r['ci_width_status']='skipped_legal_infinity' if inf else 'finite'
        r['ci_width']=None if inf else bootstrap(w2-w1,J)
        if inf: r['infinity_detail']=infinity_detail(by)
    elif kind=='H3':
        n,rr=vec('N','IS'),vec('R','IS'); ni=bool(np.isposinf(n).any()); ri=bool(np.isposinf(rr).any())
        r['N_IS_infinite']=ni; r['R_IS_infinite']=ri
        r['ci_IS_status']='skipped_legal_infinity' if ni or ri else 'finite'
        r['ci_IS_diff']=None if ni or ri else bootstrap(rr-n,J)
        if ni or ri: r['infinity_detail']=infinity_detail(by)
    elif kind=='C':
        group=branch_primary('C',tasks,independent)
        r['ci_group_cover']=bootstrap(100*(vec('R','C',group)-vec('R','C')),J)
        r['ci_group_gain']=bootstrap(100*((vec('R','C',group)-vec('N','C',group))-(vec('R','C')-vec('N','C'))),J)
    status,codes=judge(kind,r)
    assert 'U_AUDIT' not in codes and 'U_DELIVERY' not in codes
    snapshot={'kind':kind,'record':r,'status':status,'codes':codes,'locked_bjt':now(),'object_gates':{f'O{i}':'PASS' for i in range(1,8)},'evidence':{'input_manifest_sha256':filehash(ROOT/'locks/input_audit.json'),'split_manifest_sha256':filehash(ROOT/'locks/splits.json'),'execution_lock_sha256':filehash(ROOT/'locks/execution.json'),'required_task_row_hashes':{t['key']:read(ROOT/'results'/f'{t["key"]}.json')['rows_sha256'] for t in tasks if t['key'] in independent and t['mono'] and t['p']==93 and t['n']=='all' and t['branch'] in ({'A'} if kind=='A' else {'B','C'} if kind=='C' else {'B'})}}}
    atomic_json(path,snapshot)
    return snapshot
