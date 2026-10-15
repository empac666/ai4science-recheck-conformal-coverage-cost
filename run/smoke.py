from common import *
from numerics import check_examples
import time, resource, gc
from threadpoolctl import threadpool_info

def main():
    check_examples()
    rng=np.random.RandomState(300001)
    X=rng.normal(size=(515,8)).astype(np.float64); y=rng.normal(size=515).astype(np.float64)
    t=time.monotonic(); c=time.process_time()
    for q in (.05,.95):
        m=estimator(q,MONO); m.fit(X,y); assert m.n_iter_==250
        del m
    del X,y; gc.collect()
    info=threadpool_info(); assert all(x['num_threads']==1 for x in info)
    atomic_json(ROOT/'locks/smoke.json',{'status':'PASS','namespace':'smoke','key':'smoke:300001:515x8:synthetic','seed':300001,'shape':[515,8],'synthetic_finite_labels':True,'fits':2,'models_discarded':True,'scientific_metrics_generated':False,'wall_seconds':time.monotonic()-t,'cpu_seconds':time.process_time()-c,'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'environment':versions(),'threadpools':[{k:x.get(k) for k in ('internal_api','num_threads','version','architecture')} for x in info],'bjt':now()})
    print('SYNTHETIC_SMOKE_PASS',flush=True)
if __name__=='__main__': main()
