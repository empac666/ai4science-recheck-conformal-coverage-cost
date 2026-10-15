from common import *
from audit import val,bootstrap

def distribution(v):
    a=np.array(v,dtype=float); finite=a[np.isfinite(a)]
    out={'seed_count':len(a),'finite_seed_count':len(finite),'positive_infinity_seed_count':int(np.isposinf(a).sum()),'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'std_ddof1':float(a.std(ddof=1)) if np.isfinite(a).all() and len(a)>1 else {'value':None,'state':'missing/error','reason':'undefined with legal infinity or fewer than two seeds'},'finite_subset':{'n':len(finite),'mean':float(finite.mean()) if len(finite) else None,'min':float(finite.min()) if len(finite) else None,'max':float(finite.max()) if len(finite) else None}}
    return out

def finish(tasks,done,independent,objects,fit_records,start):
    import time
    assert len(done)==len(tasks)==7344 and len({t['key'] for t in tasks})==7344
    assert all(z['status'] in ('COMPLETED','not_applicable_capacity') for z in done.values())
    assert sum(z['status']=='COMPLETED' for z in fit_records)==236 and len(fit_records)<=472
    aliases=read(ROOT/'locks/aliases.json')
    for alias,key in aliases.items():
        assert key in done
        meta=done[key]; assert meta['rows_sha256']==filehash(ROOT/'rows'/f'{key}.npz')
    # Re-open every independent summary and validate the content hash/state against its live source.
    for t in tasks:
        key=t['key']
        if key not in independent: continue
        z=read(ROOT/'independent'/f'{key}.json')
        assert clean(independent[key])==z
        assert done[key]['rows_sha256']==filehash(ROOT/'rows'/f'{key}.npz')
    groups={}
    for t in tasks:
        z=done[t['key']]
        if z['status']!='COMPLETED': continue
        cfg={k:t[k] for k in ('namespace','branch','mono','p','n','r','method','boundary','legacy','rho')}
        g=json.dumps(cfg,sort_keys=True)
        groups.setdefault(g,[]).append((t['seed'],z))
    summary=[]
    J=np.load(ROOT/'locks/bootstrap_J.npy',allow_pickle=False)
    for cfg,items in groups.items():
        cfg=json.loads(cfg); items.sort(key=lambda x:x[0]); mets=[z['metrics'] for s,z in items]
        fields=['C','W_raw','W_set','W','IS','E','eta_cal','eta_test','F_row','corridor_mean_width','base_cross_rate','preclip_empty_rate','outside_corridor_rate','full_corridor_rate','clip_active_rate','endpoint_violation_rate']
        d={'configuration':cfg,'seed_ids':[s for s,z in items],'metrics':{f:distribution([m[f] for m in mets]) for f in fields},'F_cal':sum(m['F_cal'] for m in mets)/len(mets)}
        total=sum(m['test_n'] for m in mets)
        d['pooled']={f:sum(m[f]*m['test_n'] for m in mets)/total for f in fields}
        d['test_row_count']=total
        d['per_seed']=[{'seed':s,'metrics':z['metrics'],'task_key':z['task']['key']} for s,z in items]
        if cfg['branch'] in ('B','C') and len(items)==20:
            d['bootstrap']={}
            for f in ('C','W','IS'):
                a=np.array([m[f] for m in mets]); inf=not np.isfinite(a).all()
                d['bootstrap'][f]={'ci_status':'skipped_legal_infinity','ci':None,'infinite_seed_ids':[s for (s,z),v in zip(items,a) if math.isinf(v)]} if inf else {'ci_status':'finite','ci':bootstrap(a,J)}
        summary.append(d)
    atomic_json(ROOT/'branch_summaries.json',summary)
    # Paired descriptive differences; no infinity arithmetic, no bootstrap after infinity.
    paired=[]
    for cfg,items in groups.items():
        c=json.loads(cfg)
        if c['method']!='N': continue
        other=c.copy(); other['method']='R'; ri=groups.get(json.dumps(other,sort_keys=True)); assert ri is not None
        ni=sorted(items); ri=sorted(ri); assert [s for s,z in ni]==[s for s,z in ri]
        dd={'configuration':{k:v for k,v in c.items() if k!='method'},'seed_ids':[s for s,z in ni],'differences':{}}
        for f in ('C','W','W_raw','IS'):
            n=np.array([z['metrics'][f] for s,z in ni]); r=np.array([z['metrics'][f] for s,z in ri])
            if not(np.isfinite(n).all() and np.isfinite(r).all()):
                dd['differences'][f]={'ci_status':'skipped_legal_infinity','value':None,'N_infinite_seed_ids':[s for (s,z),v in zip(ni,n) if math.isinf(v)],'R_infinite_seed_ids':[s for (s,z),v in zip(ri,r) if math.isinf(v)]}; continue
            diff=(r-n)*(100 if f=='C' else 1)
            desc={'mean':float(diff.mean()),'per_seed':diff.tolist(),'unit':'pp' if f=='C' else 'GPa' if c['branch']=='FRP' else 'MPa'}
            if c['branch'] in ('B','C') and len(ni)==20: desc.update(ci_status='finite',ci=bootstrap(diff,J))
            dd['differences'][f]=desc
        mn=np.mean([z['metrics']['W'] for s,z in ni]); mr=np.mean([z['metrics']['W'] for s,z in ri])
        dd['relative_W_increase_percent']=100*(mr/mn-1) if math.isfinite(mn) and math.isfinite(mr) and mn>0 else {'value':None,'state':'missing/error','reason':'undefined ratio'}
        paired.append(dd)
    atomic_json(ROOT/'paired_summaries.json',paired)
    # Independent mechanism and combination state does not alter any primary object.
    A,B=objects['A'],objects['B']
    combination='失败' if '失败' in (A['status'],B['status']) else '成功' if A['status']==B['status']=='成功' else '不确定'
    atomic_json(ROOT/'combination.json',{'status':combination,'A':{'status':A['status'],'codes':A['codes']},'B':{'status':B['status'],'codes':B['codes']}})
    ledger={'T1':'PASS','T2':'PASS','T3':'PASS' if read(ROOT/'locks/frp_reference.json')['match'] else 'FAIL','T4':'PENDING_SUPERVISOR_FINAL_ACCOUNTING','planned':7344,'completed':sum(z['status']=='COMPLETED' for z in done.values()),'not_applicable_capacity':sum(z['status']=='not_applicable_capacity' for z in done.values()),'missing':[],'unique_fit_count':236,'fit_attempts':len(fit_records),'compute_worker_wall_seconds':time.monotonic()-start,'objects_atomic_locked':list(objects),'integrity':'未完成；等待独立监督最终 T4 账本'}
    atomic_json(ROOT/'round_audit.json',ledger)
    text=['任务完整性：等待独立监督完成 T4；A 原数字复现、B 新划分稳定性、H3 区间评分、C 配方诊断依次如下。','', '|对象|三态|原因码|','|---|---|---|']
    for k in ('A','B','H3','C'):
        o=objects[k]; text.append(f'|{k}|{o["status"]}|{", ".join(o["codes"])}|')
        if 'E_LOCKED_NONPAPER' in o['codes']: text.append('仅在事前锁定的所报补丁版本组内成立；与论文环境不同（E_LOCKED_NONPAPER）。')
        if 'F_NONPOS_B_WIDTH' in o['codes']: text.append('宽度正代价未成立。')
    text+=['',f'组合：{combination}；A：{A["status"]}，{A["codes"]}；B：{B["status"]}，{B["codes"]}。','H3 始终独立并列；无限诊断及全部种子明细见 objects/H3.json 和 objects/B.json。','主表与逐种子完整指标见 branch_summaries.json，配对差与无穷短路见 paired_summaries.json；逐行结果为 rows/*.npz，无损 float64。','D 桥接、旧秩/全局边界泄漏归因及全压力矩阵独立保存，不改变主对象终态。','固定数据上的随机划分稳定性不代表独立实测材料总体误差；Concrete 上界为经验边界，FRP 为合成诊断。']
    (ROOT/'report.md').write_text('\n'.join(text)+'\n')
