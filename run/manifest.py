from common import *

def make_manifest():
    data,unique,gids,_=dataset(); tasks=[]; fits=[]; splits={}
    def add(branch,s,mono,p=93,n='all',r=None,namespace='formal',rho=None,boundary='train',legacy=False,methods=('U','N','R')):
        for method in methods:
            t=dict(namespace=namespace,branch=branch,seed=s,mono=mono,p=p,n=n,r=r,method=method,boundary=boundary,legacy=legacy,rho=rho)
            t['key']=sha(json.dumps(t,sort_keys=True).encode()); tasks.append(t)
    for branch,seeds in [('A',range(6)),('B',range(100,120)),('C',range(100,120)),('D',range(6)),('FRP',range(2000,2020))]:
        for s in seeds:
            if branch=='FRP': tr,ca,te=row_split(np.arange(1500),s-2000)
            else: tr,ca,te=split(branch,s,unique,gids)
            sk=f'{branch}-{s}'; splits[sk]={'train':tr.tolist(),'cal':ca.tolist(),'test':te.tolist(),'train_sha256':ihash(tr),'cal_sha256':ihash(ca),'test_sha256':ihash(te)}
            for mono in ([True] if branch in ('D','FRP') else [True,False]):
                for q in [.05,.95]:
                    fits.append({'namespace':'formal','branch':branch,'seed':s,'mono':mono,'quantile':q,'data_sha256':DATA_HASH if branch!='FRP' else {'X':ahash(frp(s)[0]),'y':ahash(frp(s)[1])},'split_sha256':{k:v for k,v in splits[sk].items() if k.endswith('sha256')},'params':PARAMS})
    # Required order: primary objects before diagnostics.
    for branch,seeds in [('A',range(6)),('B',range(100,120)),('C',range(100,120)),('D',range(6))]:
        for s in seeds: add(branch,s,True)
    for branch,seeds in [('A',range(6)),('B',range(100,120)),('C',range(100,120))]:
        for s in seeds:
            for p in GRID:
                if p!=93: add(branch,s,True,p)
    for branch,seeds in [('A',range(6)),('B',range(100,120)),('C',range(100,120))]:
        for s in seeds:
            for p in GRID: add(branch,s,False,p,methods=('U',) if branch=='A' else ('U','N','R'))
    for branch in ('B','C'):
        for s in range(100,120):
            for r in range(10):
                for n in (25,50,100,200): add(branch,s,True,n=n,r=r)
    for s in range(6):
        add('A',s,True,namespace='formal_attribution',legacy=True)
        add('A',s,True,namespace='formal_attribution',boundary='global93')
        add('A',s,True,namespace='formal_attribution',boundary='loader_default')
    for d in range(2000,2020):
        for rho in RHOS: add('FRP',d,True,p=None,rho=rho,boundary='mechanism')
    assert len(tasks)==7344 and len(fits)==236
    assert len({t['key'] for t in tasks})==7344
    for f in fits: f['key']=sha(json.dumps(f,sort_keys=True).encode())
    atomic_json(ROOT/'locks/splits.json',splits)
    atomic_json(ROOT/'locks/tasks.json',tasks)
    atomic_json(ROOT/'locks/fits.json',fits)
