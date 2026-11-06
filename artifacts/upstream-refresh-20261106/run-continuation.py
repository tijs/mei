import json, os, subprocess, time, re
from pathlib import Path
root=Path('/Users/tijs/projects/vmlx-mei-upstream-sync-20261106')
out=Path('/tmp/mei-upstream-regression-20261106')
suites=['SaveTests', 'QuantizationTests', 'JSONSchemaLogitProcessorTests', 'JSONSchemaGrammarTests', 'UnusedPromptSnapshotTests', 'MiniMaxM3PromptSnapshotViewTests', 'RotatingPrefixReplayTests', 'MTPRuntimeFocusedTests', 'RawTextGenerationTests', 'TokenIteratorAdditionalProcessorTests', 'BatchEngineGrowingChatCacheSourceTests']
env=os.environ.copy();env.update(DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer',VMLX_TEST_METALLIB_PATH=str(root/'.build/out/Products/Debug/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib'),VMLX_ENABLE_UNSAFE_COMPILE='1')
rows=[]
for suite in suites:
 print('START',suite,flush=True);start=time.monotonic()
 with (out/f'{suite}.log').open('w') as stream:
  try:
   result=subprocess.run(['swift','test','--skip-build','--filter',suite],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
   status=result.returncode
  except subprocess.TimeoutExpired: status='timeout'
 contents=(out/f'{suite}.log').read_text()
 counts=[int(v) for v in re.findall(r'Executed (\d+) tests?',contents)]
 counts += [int(v) for v in re.findall(r'Test run with (\d+) tests?.*passed',contents)]
 if status==0 and max(counts,default=0)==0: status='no-tests'
 rows.append({'suite':suite,'exit':status,'seconds':round(time.monotonic()-start,3),'log':str(out/f'{suite}.log')})
 (out/'summary-continuation.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('DONE',suite,status,flush=True)
 if status!=0: break
raise SystemExit(0 if all(r['exit']==0 for r in rows) and len(rows)==len(suites) else 1)
