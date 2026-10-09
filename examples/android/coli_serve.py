#!/data/data/com.termux/files/usr/bin/python
"""Phone-local, locked, on-demand Colibri service. Never exposes the LAN."""
import fcntl,json,os,pathlib,signal,subprocess,sys,time,urllib.request
HOME=pathlib.Path(os.environ.get('COLI_SERVE_HOME','/data/data/com.termux/files/home'))
repo=pathlib.Path(os.environ.get('COLI_REPO',HOME/'projects/colibri'))
model=pathlib.Path(os.environ.get('COLI_MODEL',HOME/'models/huihui-claude-opus-qwen36-i4-gs64'))
state=HOME/'.local/state/colibri';state.mkdir(parents=True,exist_ok=True);os.chmod(state,0o700)
ID=os.environ.get('COLI_MODEL_ID','huihui-qwen3.6-35b-a3b-colibri-int4')
url='http://127.0.0.1:8080/v1/models'
def ready():
 try:
  with urllib.request.urlopen(url,timeout=2) as r:return any(x['id']==ID for x in json.load(r)['data'])
 except Exception:return False
def owned(pid):
 try:return str(repo/'c/openai_server.py').encode() in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
 except OSError:return False
with (state/'start.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 try:pid=int((state/'server.pid').read_text())
 except (OSError,ValueError):pid=0
 action=sys.argv[1] if len(sys.argv)>1 else 'start'
 if action=='status':
  print('ready: '+url if ready() else 'not ready');sys.exit(0 if ready() else 1)
 if action=='stop':
  if pid and owned(pid):
   os.killpg(pid,signal.SIGTERM)
   for _ in range(50):
    if not owned(pid):break
    time.sleep(.1)
  print('stop requested');sys.exit(0)
 if action not in ('start','ensure'):raise SystemExit('Usage: coli_serve [start|ensure|status|stop]')
 if ready():sys.exit(0)
 child=None
 if not pid or not owned(pid):
  env=dict(os.environ)
  for key in ('SERVE','PPL','DUMP','COLI_KEEP_F32','LD_PRELOAD'):env.pop(key,None)
  env.update(HOME=str(HOME),PATH='/data/data/com.termux/files/usr/bin:/system/bin',
   COLI_VULKAN='1',COLI_HEXAGON='0',COLI_CUDA='0',COLI_DENSE_I8='1',COLI_DENSE_BITS='8',COLI_DENSE_IDOT='0',
   QWEN_VK_DENSE_MB='2048',QWEN_EMBED_STREAM='1',Q36_MAXT='8192',OMP_NUM_THREADS='4',OMP_WAIT_POLICY='PASSIVE',
   COLI_NO_OMP_TUNE='1',PILOT='0',HOT='0',COLI_VK_SHADERS=str(repo/'c/shaders/qmatmul.spv'))
  log=(state/'server.log').open('ab');os.chmod(state/'server.log',0o600)
  cmd=['/data/data/com.termux/files/usr/bin/python',str(repo/'c/openai_server.py'),'--model',str(model),
       '--engine',str(repo/'c/qwen36'),'--host','127.0.0.1','--port','8080','--model-id',ID,
       '--cap','8','--max-tokens','1024','--max-queue','2','--queue-timeout','3600']
  child=subprocess.Popen(cmd,env=env,cwd=repo/'c',stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,close_fds=True)
  pid=child.pid;(state/'server.pid').write_text(str(pid)+'\n');log.close()
 print('Starting local Adreno Colibri server...',file=sys.stderr)
 for _ in range(180):
  if ready():print('Colibri ready: '+url,file=sys.stderr);break
  if (child is not None and child.poll() is not None) or (child is None and not owned(pid)):
   raise SystemExit('Colibri exited; see '+str(state/'server.log'))
  time.sleep(1)
 else:raise SystemExit('Colibri startup timed out; see '+str(state/'server.log'))
