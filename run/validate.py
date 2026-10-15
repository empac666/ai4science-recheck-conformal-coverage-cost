"""Synthetic-only semantic validation. No estimator fits, no registered labels."""
from common import *
from numerics import *
from audit import recompute
from judge import *

def run():
    assert check_examples()
    rng=np.random.RandomState(330001); passed=0
    for n in (8,9,10,19,20,25,251):
        for mode in ('normal','cross','outside','infinity','negative'):
            ca=rng.normal(size=n); cb=ca+rng.uniform(.1,2,size=n); yc=rng.normal(size=n)
            if mode=='negative': yc=(ca+cb)/2
            if mode=='outside': yc[:max(1,n//3)]=100
            if mode=='infinity': yc[:]=100
            ql=rng.normal(size=30); qh=ql+(.5 if mode!='cross' else -.5)
            yt=rng.normal(size=30); L=np.zeros(30); U=np.full(30,3.)
            raw=np.maximum(ca-yc,yc-cb); out=(yc<0)|(yc>3)
            pred={'cal_ids':np.arange(n),'cal_y':yc,'cal_lo':ca,'cal_hi':cb,'test_ids':np.arange(30),'test_group':np.full(30,-1),'test_lo':ql,'test_hi':qh,'test_y':yt}
            for method in ('U','N','R'):
                scores=raw.copy()
                if method=='R': scores[out]=math.inf
                Q,k=threshold(scores)
                arrays=evaluate(ql,qh,yt,L,U,Q,method); arrays.update(original_id=pred['test_ids'],anonymous_group=pred['test_group'])
                m=summarize(arrays,Q,k,out.mean(),n,None)
                recompute({'branch':'A','method':method,'legacy':False},{'Q':Q,'k':k,'upper_bound':3.,'metrics':m},arrays,pred,np.arange(n))
                passed+=1
    # Feature-dependent FRP corridors, synthetic labels and quantiles (no registered generator seed).
    X=rng.uniform(.2,.4,(30,5)); X[:,1]=100.; X[:,2]=5.
    XC=rng.uniform(.2,.4,(25,5)); XC[:,1]=100.; XC[:,2]=5.
    lc,uc=bounds(XC,.965); L,U=bounds(X,.965)
    yc=(lc+uc)/2; yt=(L+U)/2; qlc=yc-1.; qhc=yc+1.; ql=yt-1.; qh=yt+1.
    pred={'cal_ids':np.arange(25),'cal_y':yc,'cal_lo':qlc,'cal_hi':qhc,'cal_X':XC,'test_ids':np.arange(30),'test_group':np.full(30,-1),'test_lo':ql,'test_hi':qh,'test_y':yt}
    for method in ('U','N','R'):
        Q,k=threshold(np.maximum(qlc-yc,yc-qhc)); arrays=evaluate(ql,qh,yt,L,U,Q,method)
        arrays.update(original_id=pred['test_ids'],anonymous_group=pred['test_group'])
        m=summarize(arrays,Q,k,0.,25,None)
        recompute({'branch':'FRP','method':method,'legacy':False,'rho':.965},{'Q':Q,'k':k,'upper_bound':None,'metrics':m},arrays,pred,np.arange(25))
        passed+=1
    def base(k): return {'delivered':True,'object_audit_pass':True,'locked_environment_match':True,'exact_environment':True,'units':UNITS[k]}
    a=base('A')|{'A_vector':[85.3,89.9,4.6,18.6,23.3,4.7]}
    assert judge('A',a)==('成功',['S_A'])
    for x in (None,[],True,4,{}, {'delivered':True}): assert judge('A',x)[0]=='不确定'
    for x in ([],[1]*5,[1]*7,[1,2,3,4,5,float('nan')],[1,2,3,4,5,float('inf')]): assert judge('A',a|{'A_vector':x})[0]=='不确定'
    assert judge('A',a|{'exact_environment':False})[1]==['U_ENV']
    b=base('B')|{'ci_R_cover':[87.,93.],'ci_gain':[.01,1.],'ci_width':[.01,1.],'any_N_R_width_infinite':False,'ci_width_status':'finite'}
    assert judge('B',b)==('成功',['S_B'])
    assert judge('B',b|{'ci_R_cover':[85.,86.9]})[1]==['F_LOW_B_C']
    assert judge('B',b|{'ci_R_cover':[93.1,94.]})[1]==['F_HIGH_B_C']
    assert judge('B',b|{'ci_gain':[-1.,0.]})[1]==['F_NONPOS_B_GAIN']
    assert judge('B',b|{'ci_width':[-1.,0.]})[1]==['F_NONPOS_B_WIDTH']
    assert judge('B',b|{'ci_width':[1.,0.]})[1]==['U_AUDIT']
    assert judge('B',b|{'ci_gain':[0.,1.]})[1]==['U_CI_BOUNDARY']
    h=base('H3')|{'R_IS_infinite':False,'N_IS_infinite':False,'ci_IS_diff':[-2.,-1.],'ci_IS_status':'finite'}
    assert judge('H3',h)[1]==['S_IS_IMPROVE']
    assert judge('H3',h|{'ci_IS_diff':[1.,2.]})[1]==['F_IS_WORSE']
    assert judge('H3',h|{'ci_IS_diff':[0.,2.]})[1]==['U_IS_ZERO']
    c=base('C')|{'ci_group_cover':[-2.,0.],'ci_group_gain':[-2.,0.]}
    assert judge('C',c)[1]==['S_GROUP_NO_LARGE_DROP']
    assert judge('C',c|{'ci_group_cover':[-4.,-2.1]})[1]==['F_GROUP_DEGRADE']
    assert judge('C',c|{'ci_group_cover':[-4.,-2.]})[1]==['U_GROUP_BOUNDARY']
    # Full infinity provenance contracts, including all 20 registered seeds with zero counts.
    from audit import infinity_detail
    by={(s,m):{'W':math.inf if s==100 and m=='N' else 1.,'IS':math.inf if s==100 and m=='N' else 2.,'test_n':252,'empty_rows':1 if s==100 and m=='N' else 0,'near_zero_empty_rows':0} for s in range(100,120) for m in ('N','R')}
    detail=infinity_detail(by)
    assert judge('B',b|{'any_N_R_width_infinite':True,'ci_width_status':'skipped_legal_infinity','ci_width':None,'infinity_detail':detail})[1]==['U_EMPTY_WIDTH']
    assert judge('H3',h|{'N_IS_infinite':True,'ci_IS_status':'skipped_legal_infinity','ci_IS_diff':None,'infinity_detail':detail})[1]==['U_EMPTY_N']
    assert judge('B',b|{'exact_environment':False})[1]==['S_B','E_LOCKED_NONPAPER']
    atomic_json(ROOT/'locks/semantic_validation.json',{'status':'PASS','synthetic_interval_cases':passed,'judge_contract_and_boundary_cases':'PASS','registered_labels_read':False,'fits':0,'bjt':now()})
    print('SYNTHETIC_SEMANTIC_VALIDATION_PASS',passed)
if __name__=='__main__': run()
