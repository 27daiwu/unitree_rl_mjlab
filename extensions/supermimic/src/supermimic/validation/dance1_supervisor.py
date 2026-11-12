"""One absolute deadline for syntax/import and existing dance1 CPU validation."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]

def run_process(command, deadline, log_path, env):
    remaining=deadline-time.time()
    if remaining<=0:
        return {'status':'REFUSED_EXPIRED','launched':False,'returncode':None}
    with Path(log_path).open('w') as log:
        # Recheck immediately before Popen. Never clamp expired time to a positive value.
        if deadline-time.time()<=0:
            return {'status':'REFUSED_EXPIRED','launched':False,'returncode':None}
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            remaining=deadline-time.time()
            if remaining<=0:
                raise subprocess.TimeoutExpired(command,0)
            code=process.wait(timeout=remaining)
            return {'status':'EXITED','launched':True,'returncode':code}
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            code=process.wait()
            return {'status':'TIMEOUT_KILLED','launched':True,'returncode':code}

def main(out):
    out=Path(out);start=json.loads((out/'clock_start.json').read_text())['start_epoch'];deadline=start+178
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(ROOT/'extensions/supermimic/src')+':'+str(ROOT))
    result={'absolute_deadline_epoch':deadline,'start_epoch':start,'stages':{},'status':'RUNNING'}
    def save():(out/'supervisor_result.json').write_text(json.dumps(result,indent=2)+'\n')
    with patch('subprocess.Popen') as popen:
        denied=run_process(['must-never-launch'],time.time()-1,out/'must_not_exist.log',env)
        assert denied['status']=='REFUSED_EXPIRED' and not denied['launched']
        popen.assert_not_called()
    result['expired_deadline_regression']={'status':'PASS','Popen_calls':0};save()
    base=ROOT/'extensions/supermimic'
    files=[base/'src/supermimic/validation/dance1_cpu.py',base/'src/supermimic/validation/cpu_interface_helpers.py',base/'tests/test_cpu_interface_helpers.py',Path(__file__)]
    syntax="import ast; from pathlib import Path; [ast.parse(Path(p).read_text(),filename=p) for p in "+repr([str(p) for p in files])+"]; import supermimic.validation.dance1_cpu; import supermimic.validation.cpu_interface_helpers; print('PASS_SYNTAX_IMPORT')"
    python=str(base/'.venv/bin/python')
    result['stages']['syntax_import']=run_process([python,'-c',syntax],deadline,out/'syntax_import.log',env);save()
    if result['stages']['syntax_import'].get('returncode')==0:
        result['stages']['existing_worker']=run_process([python,'-m','supermimic.validation.dance1_cpu','--out',str(out)],deadline,out/'run.log',env);save()
    worker=out/'result.json'
    result['status']=json.loads(worker.read_text())['status'] if worker.exists() else 'BLOCKED_NO_WORKER_RESULT'
    result['checks_elapsed_s']=time.time()-start;result['checks_budget_passed']=result['checks_elapsed_s']<=180
    save();print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);main(parser.parse_args().out)
