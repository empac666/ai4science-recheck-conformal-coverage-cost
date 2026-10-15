"""Independent resource supervisor; stdlib + psutil, no numerical workload."""
import os,sys,time,json,hashlib,datetime,signal,subprocess,resource
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parent; BASE=ROOT.parent
LIMIT_WALL=21600.; LIMIT_CPU=86400.; STOP_WALL=21580.; STOP_CPU=86320.; LIMIT_RSS=8*1024**3

def now(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def atomic(p,x):
    b=json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2).encode(); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_bytes(b); os.replace(tmp,p)
def read(p): return json.loads(p.read_text())
def log(event):
    with (ROOT/'logs/formal.log').open('a') as f: f.write(json.dumps({'bjt':now(),**event},ensure_ascii=False)+'\n')
def worker_limits():
    # Aggregate RSS monitored separately, address-space limit prevents large worker allocations.
    resource.setrlimit(resource.RLIMIT_AS,(int(7.5*1024**3),int(7.5*1024**3)))
    resource.setrlimit(resource.RLIMIT_CPU,(86320,86325))
    os.umask(0o077)
def main():
    os.chdir(BASE); os.umask(0o077)
    assert not (ROOT/'launch.json').exists(),'formal launch cannot reset the clock'
    selfp=psutil.Process(); start=time.monotonic()-max(0,time.time()-selfp.create_time()); begin=now()
    raw=(ROOT/'logs/runtime.raw.log').open('ab',buffering=0)
    child=subprocess.Popen([sys.executable,'run/runner.py'],cwd=BASE,stdout=raw,stderr=raw,start_new_session=True,preexec_fn=worker_limits)
    launch={'status':'RUNNING','supervisor_pid':os.getpid(),'worker_pid':child.pid,'worker_pgid':child.pid,'started_bjt':begin,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_monotonic':start,'log':'run/logs/formal.log','relative_command':'python run/supervise.py','worker_limit':1,'threads_per_worker':1,'wall_limit_seconds':21600,'cpu_limit_seconds':86400,'memory_limit_bytes':LIMIT_RSS,'execution_lock_sha256':hashlib.sha256((ROOT/'locks/execution.json').read_bytes()).hexdigest()}
    atomic(ROOT/'launch.json',launch)
    (BASE/'run_status.md').write_text(f'STATUS: RUNNING\n\n北京时间开始：{begin}\n监督 PID：{os.getpid()}；正式工作 PID/进程组：{child.pid}\n日志：run/logs/formal.log\n资源账本：run/resources.jsonl；启动记录：run/launch.json\n冻结 SHA-256：{hashlib.sha256((BASE/"prereg_R13_FROZEN.md").read_bytes()).hexdigest()}\n预算：墙钟 ≤21600 秒、总 CPU ≤86400 秒、工作进程 1（上限4）、单线程 BLAS/OpenMP、内存 ≤8 GiB。\n独立监督每 0.5 秒检查；21580 秒或 86320 CPU 秒触发整组 SIGTERM，5 秒后 SIGKILL。\n合成标签烟测 PASS；源码、数据、FRP T3、环境、代码和任务清单已锁定。\n')
    log({'event':'STARTED','supervisor_pid':os.getpid(),'worker_pid':child.pid})
    seen={}; peak=0; reason=None; last_heartbeat=None; last_progress=time.monotonic(); previous_io=None
    while True:
        elapsed=time.monotonic()-start; cpu_self=sum(selfp.cpu_times()[:2]); live=[]
        try:
            cp=psutil.Process(child.pid); live=[cp]+cp.children(recursive=True)
        except psutil.NoSuchProcess: pass
        rss=selfp.memory_info().rss; io_total=0
        for p in live:
            try:
                ct=p.cpu_times(); ident=(p.pid,p.create_time()); seen[ident]=max(seen.get(ident,0),ct.user+ct.system)
                rss+=p.memory_info().rss; io=p.io_counters(); io_total+=io.read_bytes+io.write_bytes
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        peak=max(peak,rss); cpu_children=sum(seen.values()); total=cpu_self+cpu_children
        try: fit_cpu=read(ROOT/'cpu_classification.json')['fit_cpu_seconds']
        except (FileNotFoundError,json.JSONDecodeError): fit_cpu=0.
        entry={'bjt':now(),'wall_seconds':elapsed,'cpu_supervisor_seconds':cpu_self,'cpu_fit_seconds':fit_cpu,'cpu_other_seconds':max(0,cpu_children-fit_cpu),'cpu_total_seconds':total,'rss_bytes':rss,'peak_rss_bytes':peak,'live_compute_processes':len(live)}
        with (ROOT/'resources.jsonl').open('a') as f: f.write(json.dumps(entry)+'\n')
        code=child.poll()
        if code is not None: break
        try: hb=read(ROOT/'heartbeat.json')['monotonic']
        except (FileNotFoundError,json.JSONDecodeError): hb=start
        current=(cpu_children,io_total)
        if hb!=last_heartbeat or current!=previous_io:
            last_progress=time.monotonic(); last_heartbeat=hb; previous_io=current
        if elapsed>=STOP_WALL: reason='wall_termination_checkpoint'
        elif total>=STOP_CPU: reason='cpu_termination_checkpoint'
        elif rss>=LIMIT_RSS: reason='aggregate_memory_limit'
        elif len(live)>4: reason='worker_process_limit'
        elif time.monotonic()-hb>=600 and time.monotonic()-last_progress>=600: reason='recorded_hang_no_heartbeat_cpu_disk_progress'
        if reason:
            log({'event':'SIGTERM','reason':reason,'wall_seconds':elapsed,'cpu_seconds':total})
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                child.wait(timeout=5)
            break
        time.sleep(.5)
    code=child.wait(); raw.close(); usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    # wait4 accounting includes reaped compute processes; use max rather than adding it twice.
    children_cpu=max(sum(seen.values()),usage.ru_utime+usage.ru_stime)
    supervisor_cpu=sum(selfp.cpu_times()[:2]); elapsed=time.monotonic()-start
    try: fit_cpu=read(ROOT/'cpu_classification.json')['fit_cpu_seconds']
    except (FileNotFoundError,json.JSONDecodeError): fit_cpu=0.
    total=children_cpu+supervisor_cpu
    good=elapsed<=LIMIT_WALL and total<=LIMIT_CPU and peak<=LIMIT_RSS and reason is None and code==0
    accounting={'status':'COMPLETED' if good else 'BLOCKED','reason':reason or ('normal_exit' if code==0 else 'worker_technical_or_audit_stop'),'ended_bjt':now(),'started_bjt':begin,'worker_exit_code':code,'wall_seconds':elapsed,'cpu_supervisor_seconds':supervisor_cpu,'cpu_fit_seconds':fit_cpu,'cpu_other_seconds':max(0,children_cpu-fit_cpu),'cpu_total_seconds':total,'peak_rss_bytes':peak,'T4':'PASS' if good else 'FAIL','retired_process_cpu_retained':True,'double_counted_cpu':False}
    # Final accounting includes the supervisor through finalization; ample checkpoints reserve the tail.
    if (ROOT/'round_audit.json').exists():
        ra=read(ROOT/'round_audit.json'); ra['T4']=accounting['T4']; ra['integrity']='完成' if good and all(ra.get(k)=='PASS' for k in ('T1','T2','T3')) else '未完成'; atomic(ROOT/'round_audit.json',ra)
    accounting['cpu_supervisor_seconds']=sum(selfp.cpu_times()[:2]); accounting['cpu_total_seconds']=children_cpu+accounting['cpu_supervisor_seconds']; accounting['wall_seconds']=time.monotonic()-start
    atomic(ROOT/'resource_final.json',accounting)
    with (BASE/'run_status.md').open('a') as f:
        f.write(f'\nSTATUS: {accounting["status"]}\n结束北京时间：{now()}\n原因：{accounting["reason"]}\n墙钟秒：{accounting["wall_seconds"]:.6f}；总 CPU 秒：{accounting["cpu_total_seconds"]:.6f}；峰值内存字节：{peak}\n最终账本：run/resource_final.json；轮次审计：run/round_audit.json（如已交付）。\n已锁定科学对象保持原状态，不按资源或后续诊断回写。\n')
    if (ROOT/'report.md').exists():
        with (ROOT/'report.md').open('a') as f: f.write(f'\n独立监督最终记录：T4={accounting["T4"]}；轮次完整性见 round_audit.json；资源见 resource_final.json。\n')
    log({'event':'FINISHED',**accounting})
if __name__=='__main__': main()
