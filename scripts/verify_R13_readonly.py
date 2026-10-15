import sys,json,hashlib,math,collections,datetime
from pathlib import Path
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]; R=BASE/'run'; sys.path.insert(0,str(R))
import numpy as np
from numerics import threshold,evaluate,summarize
from audit import recompute,equal
from judge import judge,UNITS

def read(p): return json.loads(Path(p).read_text())
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(x):
 if isinstance(x,dict):
  if x=={'state':'positive_infinity','value':None}: return math.inf
  return {k:decode(v) for k,v in x.items()}
 if isinstance(x,list): return [decode(v) for v in x]
 return x
# Hash all scientific/runtime artifacts; omit installed dependencies and obsolete first-attempt backup.
paths=[p for p in R.rglob('*') if p.is_file() and not any(s in p.parts for s in ('venv','uv-cache','__pycache__','backup_dead_sandbox_20261015T021036Z'))]
before={str(p.relative_to(BASE)):h(p) for p in paths}
freeze=read(R/'locks/execution.json')
# Public hashes verify the sanitized package, not the unavailable original lock.
import subprocess
subprocess.run([sys.executable,str(BASE/'scripts/verify_bundle.py')],check=True)
for p,z in read(BASE/'PROVENANCE.json')['files'].items(): assert h(BASE/p)==z['public_sha256'],p
frozen_text=(BASE/'prereg_R13_FROZEN.md').read_text(); frozen_judge=frozen_text.split('```python\n')[0] if False else None
# Compare the entire executable judge block to the preregistration, ignoring file docstring/import only.
judge_text=(R/'judge.py').read_text(); assert judge_text[judge_text.index('FIELDS ='):].strip() in frozen_text
T=read(R/'locks/tasks.json'); assert len(T)==len({t['key'] for t in T})==7344
manifest=read(R/'model_manifest.json'); ledger=read(R/'fit_ledger.json'); fits=read(R/'locks/fits.json')
assert len(manifest)==118 and len(ledger)==236 and len(fits)==236
assert {x['fit_key'] for x in ledger}=={x['key'] for x in fits}
assert all(x['status']=='COMPLETED' and x['attempt']==1 and x['n_iter']==250 for x in ledger)
env=read(R/'locks/environment.json'); assert env['versions']==env['paper_versions'] and env['exact_paper_match']
from common import versions,PARAMS,dataset,clean
assert versions()==env['versions']; data,unique,gids,_=dataset()
P={}
for m in manifest:
 assert m['namespace']=='formal' and m['code_sha256']==freeze['code_fingerprint']
 assert m['params_sha256']==hashlib.sha256(json.dumps(PARAMS,sort_keys=True).encode()).hexdigest()
 assert h(R/'predictions'/f'{m["family"]}.npz')==m['predictions_sha256']
 assert m==read(R/'predictions'/f'{m["family"]}.json')
 with np.load(R/'predictions'/f'{m["family"]}.npz',allow_pickle=False) as z: P[m['family']]={k:z[k] for k in z.files}
S=read(R/'locks/splits.json'); metrics={}; statuses=collections.Counter(); last=[]
for idx,t in enumerate(T):
 key=t['key']; meta=decode(read(R/'results'/f'{key}.json')); statuses[meta['status']]+=1
 assert meta['task']==t
 if meta['status']!='COMPLETED': continue
 assert meta['model_reference']==next(m for m in manifest if m['family']==f'{t["branch"]}-{t["seed"]}-{int(t["mono"])}')
 assert h(R/'rows'/f'{key}.npz')==meta['rows_sha256']
 pred=P[f'{t["branch"]}-{t["seed"]}-{int(t["mono"])}']; ca=np.array(t['cal_ids']); pos={int(v):i for i,v in enumerate(pred['cal_ids'])}; ix=np.array([pos[int(v)] for v in ca])
 yc=pred['cal_y'][ix]; raw=np.maximum(pred['cal_lo'][ix]-yc,yc-pred['cal_hi'][ix])
 if t['branch']=='FRP':
  def bound(X):
   vf,ef,em=X[:,0],X[:,1],X[:,2]; L=1/(vf/ef+(1-vf)/em); return L,L+t['rho']*(vf*ef+(1-vf)*em-L)
  L,U=bound(pred['test_X']); lc,uc=bound(pred['cal_X'][ix])
 else:
  tr=S[f'{t["branch"]}-{t["seed"]}']['train']
  upper=float(np.percentile(data[tr,8],t['p'],method='linear')) if t['boundary']=='train' else float(np.percentile(data[:,8],93,method='linear')) if t['boundary']=='global93' else float(1.15*np.percentile(data[:,8],99,method='linear'))
  assert upper==meta['upper_bound']; L=np.zeros(len(pred['test_y']));U=np.full(len(L),upper);lc=np.zeros(len(yc));uc=np.full(len(yc),upper)
 outside=(yc<lc)|(yc>uc); scores=raw.copy()
 if t['method']=='R': scores[outside]=math.inf
 Q,k=threshold(scores,t['legacy']); equal(Q,meta['Q']);assert k==meta['k']
 a=evaluate(pred['test_lo'],pred['test_hi'],pred['test_y'],L,U,Q,t['method'])
 with np.load(R/'rows'/f'{key}.npz',allow_pickle=False) as z: rows={c:z[c] for c in z.files}
 for c,v in a.items():
  if v.dtype==bool: assert np.array_equal(v,rows[c]),(key,c)
  else: assert np.array_equal(v,rows[c]),(key,c)
 z=recompute(t,meta,rows,pred,ca)
 saved=decode(read(R/'independent'/f'{key}.json'))
 for c,v in z.items(): equal(v,saved[c])
 metrics[key]=meta['metrics']
 if idx>=7301: last.append({'index':idx,'branch':t['branch'],'seed':t['seed'],'method':t['method'],'rho':t['rho'],'completed_bjt':meta['completed_bjt']})
 if idx%1500==0: print('verified',idx,flush=True)
assert statuses=={'COMPLETED':7344}
B=decode(read(R/'branch_summaries.json')); paired=decode(read(R/'paired_summaries.json')); J=np.load(R/'locks/bootstrap_J.npy')
assert np.array_equal(J,np.random.RandomState(900001).randint(0,20,size=(10000,20)))
def ci(v): return np.percentile(np.asarray(v)[J].mean(axis=1),[2.5,97.5],method='linear').tolist()
primary={}
for b in B:
 cfg=b['configuration']
 for field,dist in b['metrics'].items(): equal(np.mean([metrics[x['task_key']][field] for x in b['per_seed']]),dist['mean'])
 if cfg['mono'] and cfg['n']=='all' and cfg['p']==93 and cfg['boundary']=='train' and not cfg['legacy']:
  primary[(cfg['branch'],cfg['method'])]=b
for p in paired:
 cfg=p['configuration']; nn=[b for b in B if b['configuration']==dict(cfg,method='N')][0];rr=[b for b in B if b['configuration']==dict(cfg,method='R')][0]
 for f,d in p['differences'].items():
  n=np.array([x['metrics'][f] for x in nn['per_seed']]);r=np.array([x['metrics'][f] for x in rr['per_seed']])
  if not(np.isfinite(n).all() and np.isfinite(r).all()): assert d['ci_status']=='skipped_legal_infinity';continue
  v=(r-n)*(100 if f=='C' else 1);equal(v.mean(),d['mean'])
  if 'ci' in d: equal(ci(v),d['ci'])
def v(branch,m,f):return np.array([x['metrics'][f] for x in primary[(branch,m)]['per_seed']])
objects={}
for kind in ('A','B','H3','C'):
 o=read(R/'objects'/f'{kind}.json');r={'delivered':True,'object_audit_pass':True,'locked_environment_match':True,'exact_environment':True,'units':UNITS[kind]}
 if kind=='A':
  cn,cr=v('A','N','C').mean(),v('A','R','C').mean();wn,wr=v('A','N','W_raw').mean(),v('A','R','W_raw').mean();r['A_vector']=[float(x) for x in [100*cn,100*cr,100*(cr-cn),wn,wr,wr-wn]]
 if kind=='B':r.update(ci_R_cover=ci(100*v('B','R','C')),ci_gain=ci(100*(v('B','R','C')-v('B','N','C'))),ci_width=ci(v('B','R','W')-v('B','N','W')),ci_width_status='finite',any_N_R_width_infinite=False)
 if kind=='H3':r.update(N_IS_infinite=False,R_IS_infinite=False,ci_IS_diff=ci(v('B','R','IS')-v('B','N','IS')),ci_IS_status='finite')
 if kind=='C':r.update(ci_group_cover=ci(100*(v('C','R','C')-v('B','R','C'))),ci_group_gain=ci(100*((v('C','R','C')-v('C','N','C'))-(v('B','R','C')-v('B','N','C')))))
 for k,value in r.items():
  if isinstance(value,list) and value and isinstance(value[0],(float,int)):equal(value,o['record'][k])
  else: assert value==o['record'][k],(kind,k)
 status,codes=judge(kind,r);assert (status,codes)==(o['status'],o['codes']);assert f'|{kind}|{status}|{", ".join(codes)}|' in (R/'report.md').read_text()
 for key,digest in o['evidence']['required_task_row_hashes'].items():assert h(R/'rows'/f'{key}.npz')==digest
 objects[kind]={'status':status,'codes':codes,'recomputed_record':r}
for p,d in before.items():assert h(BASE/p)==d,p
output={'checked_bjt':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'public_provenance_files_verified':len(read(BASE/'PROVENANCE.json')['files']),'read_only_files_verified_unchanged':len(before),'all_tasks_recomputed':len(metrics),'statuses':dict(statuses),'model_families':len(manifest),'fit_attempts':len(ledger),'objects':objects,'last_43':last,'primary':{f'{b}-{m}':z for (b,m),z in primary.items()},'paired_primary':[p for p in paired if p['configuration']['mono'] and p['configuration']['n']=='all' and p['configuration']['p']==93 and p['configuration']['boundary']=='train' and not p['configuration']['legacy']]}
import argparse
parser=argparse.ArgumentParser(description='Read-only public numerical verification; requires pinned environment.')
parser.add_argument('--output',type=Path)
args=parser.parse_args()
if args.output:
 target=args.output.resolve()
 assert not target.is_relative_to(BASE),'output must be outside the public archive'
 target.write_text(json.dumps(clean(output),ensure_ascii=False,indent=2,allow_nan=False))
print('PASS',len(metrics),'tasks;',len(before),'files unchanged;',[(k,o['status'],o['codes']) for k,o in objects.items()])
