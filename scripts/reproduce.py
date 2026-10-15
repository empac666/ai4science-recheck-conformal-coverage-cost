"""Offline reproduction in a new directory; never write into the archive."""
from pathlib import Path
import argparse,shutil,subprocess,sys,json,hashlib
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args(); dest=a.output.resolve()
assert not dest.is_relative_to(BASE) and not BASE.is_relative_to(dest),'use a separate directory outside the archive'
assert not dest.exists(),'output directory must be new'
subprocess.run([sys.executable,str(BASE/'scripts/verify_bundle.py')],check=True)
sys.path.insert(0,str(BASE/'run'))
from common import versions,PAPER
assert versions()==PAPER,'requires Python 3.12.3 and requirements.txt exactly'
dest.mkdir(); (dest/'run').mkdir(); (dest/'run/logs').mkdir(); (dest/'run/locks').mkdir()
for src in (BASE/'run').glob('*.py'):shutil.copyfile(src,dest/'run'/src.name)
shutil.copytree(BASE/'data',dest/'data')
for name in ['prereg_R13_FROZEN.md','prereg_R13_v1.2.md']:shutil.copyfile(BASE/name,dest/name)
for name in ['environment.json','frp_reference.json','input_audit.json','smoke.json','semantic_validation.json']:
 shutil.copyfile(BASE/'run/locks'/name,dest/'run/locks'/name)
# Archived upstream/FRP reference hashes are provenance, not a fresh source validation.
(dest/'REPRODUCTION_NOTICE.txt').write_text('Public reproduction, not an original formal attempt. New clocks and code fingerprint. Archived upstream/FRP reference attestations reused; original source implementations and first-attempt evidence are excluded. D-01 remains unresolved.\n')
def run(*args):subprocess.run([sys.executable,'-B',*args],cwd=dest,check=True)
run('-c',"import sys;sys.path.insert(0,'run');from manifest import make_manifest;make_manifest()")
run('run/validate.py')
run('run/freeze_execution.py')
run('run/supervise.py')
expected={'A':['S_A'],'B':['S_B'],'H3':['F_IS_WORSE'],'C':['U_GROUP_BOUNDARY']}
for kind,codes in expected.items():
 obj=json.loads((dest/'run/objects'/f'{kind}.json').read_text());assert obj['codes']==codes,(kind,obj['codes'])
assert json.loads((dest/'run/round_audit.json').read_text())['completed']==7344
print('PASS: reproduced 7344 tasks and A/B/H3/C reason codes in separate output directory.')
