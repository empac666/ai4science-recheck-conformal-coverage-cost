"""Frozen numerical configuration and lossless atomic IO."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import json, hashlib, math, datetime, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, scipy
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
COMMIT='4c5befc6305bdcdda11f4411958c6373f9028f56'
DATA_HASH='d7d8bd087f832935e902bcb2687667238cac3fe06799677a93ca2f46dce8db02'
FEATURES=['cement','slag','ash','water','superplastic','coarseagg','fineagg','age']
MONO=[1,0,0,-1,0,0,0,1]
GRID=[99,97,95,93,91,90,88,86]
RHOS=[1.,.99,.98,.965,.95,.90]
EPS=1e-8
PAPER=['3.12.3','2.4.4','1.8.0','3.0.2','1.17.1']
PARAMS=dict(loss='quantile',max_iter=250,learning_rate=.06,max_leaf_nodes=31,max_depth=None,min_samples_leaf=20,l2_regularization=1.,max_features=1.,max_bins=255,categorical_features=None,interaction_cst=None,warm_start=False,early_stopping=False,scoring='loss',validation_fraction=.1,n_iter_no_change=10,tol=1e-7,verbose=0,random_state=0)
def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(b): return hashlib.sha256(b).hexdigest()
def filehash(p): return sha(Path(p).read_bytes())
def ahash(a): return sha(np.ascontiguousarray(a,dtype='<f8').tobytes(order='C'))
def ihash(a): return sha(np.ascontiguousarray(a,dtype='<i8').tobytes(order='C'))
def clean(x):
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,np.generic): return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):
        assert x==math.inf,'illegal nonfinite serialization'
        return {'value':None,'state':'positive_infinity'}
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    return x
def atomic_json(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(clean(obj),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_bytes(b); os.replace(tmp,p)
    return sha(b)
def read(p): return json.loads(Path(p).read_text())
def versions(): return [platform.python_version(),np.__version__,sklearn.__version__,pd.__version__,scipy.__version__]
def estimator(q,mono): return HistGradientBoostingRegressor(quantile=q,monotonic_cst=mono,**PARAMS)
def frp(d):
    rs=np.random.RandomState(d)
    Vf=rs.uniform(.10,.48,1500); Ef=rs.uniform(69,700,1500); Em=rs.uniform(2,10,1500)
    radius=rs.uniform(3,12,1500); MMA=rs.uniform(0,5,1500)
    V=Vf*Ef+(1-Vf)*Em; R0=1./(Vf/Ef+(1-Vf)/Em)
    position=np.clip(.97-.04*MMA,.05,.999)
    E11=R0+position*(V-R0)
    y=np.clip(E11+rs.normal(0,.04*(V-R0)),R0,V)
    return np.column_stack([Vf,Ef,Em,radius,MMA]),y
def bounds(X,rho):
    vf,ef,em=X[:,0],X[:,1],X[:,2]
    L=1./(vf/ef+(1-vf)/em); V=vf*ef+(1-vf)*em
    return L,L+rho*(V-L)
def row_split(ids,s):
    tr,tmp=train_test_split(ids,test_size=.5,random_state=int(s),shuffle=True,stratify=None)
    ca,te=train_test_split(tmp,test_size=.5,random_state=int(s),shuffle=True,stratify=None)
    return tr,ca,te
def dataset():
    p=BASE/'data/concrete.csv'; assert filehash(p)==DATA_HASH,'O1: data hash'
    df=pd.read_csv(p,dtype=np.float64); assert list(df)==FEATURES+['strength']
    assert df.shape==(1030,9) and np.isfinite(df.to_numpy()).all(),'O1: data shape'
    du=df.drop_duplicates(subset=FEATURES+['strength'],keep='first')
    assert len(du)==1005,'O1: dedup'
    a=np.round(du[FEATURES[:-1]].to_numpy(dtype=np.float64),decimals=6)
    keys=[tuple(float(v) for v in row) for row in a]; ordered=sorted(set(keys))
    lookup={k:i for i,k in enumerate(ordered)}; assert len(ordered)==427,'O1: groups'
    gids=np.full(1030,-1,dtype=np.int64)
    gids[du.index]=[lookup[k] for k in keys]
    first={}; dup=[]
    for i,row in enumerate(df.to_numpy()):
        k=tuple(row); j=first.setdefault(k,i)
        dup.append({'original_id':i,'first_id':j,'duplicate':i!=j})
    return df.to_numpy(dtype=np.float64),du.index.to_numpy(dtype=np.int64),gids,dup
def split(branch,s,unique,gids):
    ids=np.arange(1030,dtype=np.int64) if branch=='A' else unique.copy()
    if branch!='C':
        tr,ca,te=row_split(ids,s)
        assert tuple(map(len,(tr,ca,te)))==((515,257,258) if branch=='A' else (502,251,252))
    else:
        u=np.unique(gids[unique]); np.random.RandomState(int(s)).shuffle(u)
        G=len(u); n1,n2=int(np.floor(.5*G)),int(np.floor(.75*G))
        groups=[set(u[:n1]),set(u[n1:n2]),set(u[n2:])]
        assert list(map(len,groups))==[213,107,107]
        tr,ca,te=[unique[np.isin(gids[unique],list(g))] for g in groups]
        assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    a,b,c=map(set,(tr,ca,te)); assert not(a&b or a&c or b&c) and a|b|c==set(ids)
    return tr,ca,te
