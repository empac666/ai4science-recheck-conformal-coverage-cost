"""One formal compute worker. Scientific decisions never control task execution."""
from common import *
from numerics import threshold,evaluate,summarize,check_examples
from audit import recompute,object_lock
import time, resource, traceback
from threadpoolctl import threadpool_info

class NumericFailure(RuntimeError): pass

def heartbeat(stage,detail=None):
    atomic_json(ROOT/'heartbeat.json',{'bjt':now(),'monotonic':time.monotonic(),'stage':stage,'detail':detail})

def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    os.replace(tmp,path)
    return filehash(path)

def main():
    start=time.monotonic(); assert versions()==read(ROOT/'locks/environment.json')['versions']
    frozen=read(ROOT/'locks/execution.json')
    for rel,h in frozen['files'].items(): assert filehash(BASE/rel)==h,'O1 frozen file integrity'
    check_examples(); assert all(t['num_threads']==1 for t in threadpool_info())
    data,unique,gids,_=dataset()
    tasks=read(ROOT/'locks/tasks.json'); fits=read(ROOT/'locks/fits.json'); splits=read(ROOT/'locks/splits.json')
    assert len(tasks)==7344 and len(fits)==236
    fit_lookup={(f['branch'],f['seed'],f['mono'],f['quantile']):f for f in fits}
    done={}; independent={}; model_cache={}; fit_records=[]; fit_cpu=0.; family_log=[]
    primary_counts={'A':0,'B':0,'C':0}; objects={}; capacity=0
    outdirs=['predictions','rows','results','independent','objects']
    for d in outdirs: (ROOT/d).mkdir(exist_ok=True)
    def load_model(branch,s,mono):
        nonlocal fit_cpu
        mk=(branch,s,mono)
        if mk in model_cache: return model_cache[mk]
        heartbeat('fit',{'branch':branch,'seed':s,'mono':mono})
        si=splits[f'{branch}-{s}']; tr,ca,te=[np.array(si[k],dtype=np.int64) for k in ('train','cal','test')]
        if branch=='FRP':
            X,y=frp(s); groups=np.full(len(y),-1,dtype=np.int64)
            ref=read(ROOT/'locks/frp_reference.json')
            if s==2000: assert ahash(X)==ref['reference_X_sha256'] and ahash(y)==ref['reference_y_sha256'],'T3 runtime FRP'
            atomic_json(ROOT/'predictions'/f'FRP-{s}-generation.json',{'X_sha256':ahash(X),'y_sha256':ahash(y),'seed':s,'unit':'GPa'})
            direction=[1,1,1,0,-1]
        else:
            X,y=data[:,:8],data[:,8]; groups=gids
            direction=MONO if mono else [0]*8
        a={'train_ids':tr,'cal_ids':ca,'test_ids':te,'cal_y':y[ca],'cal_X':X[ca],'test_y':y[te],'test_X':X[te],'test_group':groups[te]}
        records=[]
        for q,label in [(.05,'lo'),(.95,'hi')]:
            spec=fit_lookup[(branch,s,mono,q)]
            for attempt in (1,2):
                ct=time.process_time(); wt=time.monotonic()
                entry={'fit_key':spec['key'],'branch':branch,'seed':s,'mono':mono,'quantile':q,'attempt':attempt,'start_bjt':now(),'status':'STARTED'}
                fit_records.append(entry); atomic_json(ROOT/'fit_ledger.json',fit_records)
                try:
                    model=estimator(q,direction); model.fit(X[tr],y[tr]); assert model.n_iter_==250
                    caq=model.predict(X[ca]); teq=model.predict(X[te])
                    if not(np.isfinite(caq).all() and np.isfinite(teq).all()): raise NumericFailure('invalid base predictions')
                    a['cal_'+label]=caq; a['test_'+label]=teq
                    entry.update(status='COMPLETED',n_iter=int(model.n_iter_),full_params=model.get_params())
                    del model
                except NumericFailure:
                    entry.update(status='FAILED',error='NumericFailure')
                    if attempt==2: raise
                finally:
                    cost=time.process_time()-ct; fit_cpu+=cost
                    entry.update(cpu_seconds=cost,wall_seconds=time.monotonic()-wt,end_bjt=now())
                    atomic_json(ROOT/'fit_ledger.json',fit_records)
                    atomic_json(ROOT/'cpu_classification.json',{'fit_cpu_seconds':fit_cpu,'fit_attempts':len(fit_records)})
                if entry['status']=='COMPLETED': break
            assert len(fit_records)<=472
            records.append(spec['key'])
        assert np.isfinite(np.maximum(a['cal_lo']-a['cal_y'],a['cal_y']-a['cal_hi'])).all()
        key=f'{branch}-{s}-{int(mono)}'
        ph=save_npz(ROOT/'predictions'/f'{key}.npz',a)
        pm={'family':key,'fit_keys':records,'predictions_sha256':ph,'namespace':'formal','params_sha256':sha(json.dumps(PARAMS,sort_keys=True).encode()),'code_sha256':frozen['code_fingerprint'],'source_sha256':read(ROOT/'locks/input_audit.json')['source_sha256']}
        atomic_json(ROOT/'predictions'/f'{key}.json',pm); family_log.append(pm)
        atomic_json(ROOT/'model_manifest.json',family_log)
        model_cache[mk]=(a,pm)
        return a,pm
    for idx,t in enumerate(tasks):
        heartbeat('task',{'index':idx,'task_key':t['key']})
        branch,s,mono=t['branch'],t['seed'],t['mono']
        pred,pm=load_model(branch,s,mono)
        cal_ids=np.array(t['cal_ids'],dtype=np.int64)
        if len(cal_ids)<(t['n'] if isinstance(t['n'],int) else len(cal_ids)):
            assert isinstance(t['n'],int)
            result={'task':t,'status':'not_applicable_capacity','capacity':len(pred['cal_ids']),'cal_groups':int(len(np.unique(gids[pred['cal_ids']]))) if branch=='C' else None,'reason':'registered calibration capacity below requested n'}
            atomic_json(ROOT/'results'/f'{t["key"]}.json',result); done[t['key']]=result; capacity+=1
            continue
        calpos={int(v):i for i,v in enumerate(pred['cal_ids'])}; ci=np.array([calpos[int(v)] for v in cal_ids])
        yc=pred['cal_y'][ci]; qlc=pred['cal_lo'][ci]; qhc=pred['cal_hi'][ci]
        ql,qh,y=pred['test_lo'],pred['test_hi'],pred['test_y']
        if branch=='FRP':
            L,U=bounds(pred['test_X'],t['rho']); lc,uc=bounds(pred['cal_X'][ci],t['rho']); upper=None
        else:
            tr=np.array(splits[f'{branch}-{s}']['train'],dtype=np.int64)
            if t['boundary']=='train': upper=float(np.percentile(data[tr,8],t['p'],method='linear'))
            elif t['boundary']=='global93': upper=float(np.percentile(data[:,8],93,method='linear'))
            elif t['boundary']=='loader_default': upper=float(1.15*np.percentile(data[:,8],99,method='linear'))
            else: raise RuntimeError('unregistered boundary')
            L=np.zeros(len(y)); U=np.full(len(y),upper); lc=np.zeros(len(yc)); uc=np.full(len(yc),upper)
        raw=np.maximum(qlc-yc,yc-qhc)
        if not np.isfinite(raw).all(): raise NumericFailure('invalid raw scores')
        outside=(yc<lc)|(yc>uc)
        scores=raw.copy()
        if t['method']=='R': scores[outside]=math.inf
        Q,k=threshold(scores,t['legacy'])
        arrays=evaluate(ql,qh,y,L,U,Q,t['method'])
        arrays.update(original_id=pred['test_ids'],anonymous_group=pred['test_group'],w_state=np.isposinf(arrays['w']).astype(np.uint8),IS_state=np.isposinf(arrays['IS']).astype(np.uint8))
        cg=int(len(np.unique(gids[cal_ids]))) if branch=='C' else None
        metrics=summarize(arrays,Q,k,float(outside.mean()),len(cal_ids),cg)
        if branch=='C':
            gs=pred['test_group']; metrics['equal_recipe_cover']=float(np.mean([arrays['hit'][gs==g].mean() for g in np.unique(gs)]))
        meta={'task':t,'status':'COMPLETED','Q':Q,'k':k,'finite_count':int(np.isfinite(scores).sum()),'negative_base_score_count':int((raw<0).sum()),'upper_bound':upper,'metrics':metrics,'model_reference':pm,'cache_key':t['key'],'unit':'GPa' if branch=='FRP' else 'MPa','float_storage':'NPZ float64 lossless; JSON IEEE float round-trip','completed_bjt':now(),'Q_state':'positive_infinity' if Q==math.inf else 'finite','numeric_states':{key:('positive_infinity' if isinstance(value,(float,int)) and math.isinf(value) else 'finite') for key,value in metrics.items() if isinstance(value,(float,int))}}
        rows=ROOT/'rows'/f'{t["key"]}.npz'
        meta['rows_sha256']=save_npz(rows,arrays)
        # Reader opens committed row bytes, not in-memory production arrays.
        with np.load(rows,allow_pickle=False) as loaded:
            checked=recompute(t,meta,{k:loaded[k] for k in loaded.files},pred,cal_ids)
        atomic_json(ROOT/'independent'/f'{t["key"]}.json',checked)
        independent[t['key']]=checked
        atomic_json(ROOT/'results'/f'{t["key"]}.json',meta); done[t['key']]=meta
        if branch in primary_counts and mono and t['p']==93 and t['n']=='all' and t['namespace']=='formal':
            primary_counts[branch]+=1
            if branch=='A' and primary_counts['A']==18: objects['A']=object_lock('A',tasks,independent)
            if branch=='B' and primary_counts['B']==60:
                objects['B']=object_lock('B',tasks,independent); objects['H3']=object_lock('H3',tasks,independent)
            if branch=='C' and primary_counts['C']==60: objects['C']=object_lock('C',tasks,independent)
        if idx%50==0:
            atomic_json(ROOT/'progress.json',{'planned':7344,'completed':len(done),'not_applicable_capacity':capacity,'fit_attempts':len(fit_records),'unique_fits_completed':sum(x['status']=='COMPLETED' for x in fit_records),'stage':branch,'bjt':now()})
    from reporting import finish
    finish(tasks,done,independent,objects,fit_records,start)
    heartbeat('COMPLETED',{'tasks':len(done)})

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        evidence={'status':'BLOCKED','bjt':now(),'exception_type':type(exc).__name__,'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)],'reason':'technical or audit stop; no scientific criterion was changed'}
        atomic_json(ROOT/'failure.json',evidence)
        heartbeat('BLOCKED',evidence)
        print(json.dumps(evidence,ensure_ascii=False),flush=True)
        raise SystemExit(1)
