"""Finalize all manifests before the sole formal launch. No scientific evaluation."""
from common import *
import importlib.metadata
from threadpoolctl import threadpool_info

def main():
    assert not (ROOT/'launch.json').exists()
    assert not (ROOT/'locks/execution.json').exists()
    assert versions()==PAPER,'default exact version target unavailable'
    assert read(ROOT/'locks/smoke.json')['status']=='PASS'
    assert read(ROOT/'locks/semantic_validation.json')['status']=='PASS'
    assert read(ROOT/'locks/frp_reference.json')['match']
    # Restrict environment record to non-identifying scientific/runtime specifications.
    env=read(ROOT/'locks/environment.json')
    assert env['versions']==versions()
    env['model_params']=PARAMS
    atomic_json(ROOT/'locks/environment.json',env)
    codes={str(p.relative_to(BASE)):filehash(p) for p in sorted(ROOT.glob('*.py'))}
    code_fingerprint=sha(json.dumps(codes,sort_keys=True).encode())
    tasks=read(ROOT/'locks/tasks.json'); fits=read(ROOT/'locks/fits.json'); splits=read(ROOT/'locks/splits.json')
    source=read(ROOT/'locks/input_audit.json')['source_sha256']; _,unique,gids,_=dataset()
    for f in fits:
        f.update(code_fingerprint=code_fingerprint,source_sha256=source,environment=versions())
        f.pop('key',None); f['key']=sha(json.dumps(f,sort_keys=True).encode())
    atomic_json(ROOT/'locks/fits.json',fits)
    aliases={}
    for t in tasks:
        sp=splits[f'{t["branch"]}-{t["seed"]}']; cal=np.array(sp['cal'],dtype=np.int64)
        if isinstance(t['n'],int):
            b=0 if t['branch']=='B' else 1
            rng=np.random.RandomState(200000+1000*b+10*t['seed']+t['r'])
            if t['branch']=='B':
                cal=cal.copy(); rng.shuffle(cal)
            else:
                groups=np.unique(gids[cal]); rng.shuffle(groups)
                cal=np.concatenate([cal[gids[cal]==g] for g in groups])
            cal=cal[:t['n']]
        t.update(cal_ids=cal.tolist(),cal_id_sha256=ihash(cal),data_sha256=DATA_HASH if t['branch']!='FRP' else {'X':ahash(frp(t['seed'])[0]),'y':ahash(frp(t['seed'])[1])},split_id_sha256={k:v for k,v in sp.items() if k.endswith('sha256')},code_fingerprint=code_fingerprint,source_sha256=source,model_params_sha256=sha(json.dumps(PARAMS,sort_keys=True).encode()),rank_rule='legacy_higher' if t['legacy'] else 'integer_exact_k=(9*(n+1)+9)//10',environment=versions())
        t.pop('key',None); t['key']=sha(json.dumps(t,sort_keys=True).encode())
        if t['namespace']=='formal' and t['mono'] and t['p']==93 and t['n']=='all' and t['branch'] in ('A','B','C'):
            for label in ('main','grid_P93','cal_all'):
                aliases[f'{t["branch"]}-{t["seed"]}-{t["method"]}-{label}']=t['key']
    assert len({t['key'] for t in tasks})==7344
    assert all(t['namespace']!='smoke' and not t['key'].startswith('smoke') for t in tasks)
    atomic_json(ROOT/'locks/tasks.json',tasks); atomic_json(ROOT/'locks/aliases.json',aliases)
    J=np.random.RandomState(900001).randint(0,20,size=(10000,20))
    with (ROOT/'locks/bootstrap_J.npy').open('wb') as f: np.save(f,J,allow_pickle=False)
    files={str(p.relative_to(BASE)):filehash(p) for p in sorted(ROOT.glob('*.py'))}
    files['data/concrete.csv']=filehash(BASE/'data/concrete.csv')
    files['prereg_R13_FROZEN.md']=filehash(BASE/'prereg_R13_FROZEN.md')
    files['prereg_R13_v1.2.md']=filehash(BASE/'prereg_R13_v1.2.md')
    assert files['prereg_R13_FROZEN.md']==files['prereg_R13_v1.2.md']
    for p in sorted((ROOT/'locks').glob('*')):
        if p.is_file(): files[str(p.relative_to(BASE))]=filehash(p)
    atomic_json(ROOT/'locks/execution.json',{'files':files,'code_fingerprint':code_fingerprint,'frozen_bjt':now(),'unique_fit_limit':236,'formal_fit_attempt_limit':472,'planned_method_rows':7344,'smoke_namespace':'smoke','formal_namespace':'formal','command':'python run/supervise.py','formal_clock_not_started':True})
    print('EXECUTION_LOCKED',filehash(ROOT/'locks/execution.json'))
if __name__=='__main__': main()
