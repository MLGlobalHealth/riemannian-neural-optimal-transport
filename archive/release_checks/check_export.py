from pathlib import Path
import ast,hashlib,json,os,re,subprocess,sys,zipfile
base=Path('/private/tmp/mlgh-paper-release-20260908')
zip_path=Path(sys.argv[1]).resolve()
extracted=base/'extracted-release'
assert not extracted.exists()
with zipfile.ZipFile(zip_path) as z:
    for name in z.namelist():
        assert '..' not in Path(name).parts and not name.startswith('/')
    z.extractall(extracted)
root=extracted/'riemannian-neural-optimal-transport'
manifest=json.loads((root/'RELEASE_MANIFEST.json').read_text())
assert not any((root/name).exists() for name in ('archive','verification','website','tests'))
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
assert actual==set(manifest['files'])|{'RELEASE_MANIFEST.json'}
for name,entry in manifest['files'].items():
    content=(root/name).read_bytes()
    assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256'],name
    if name.endswith('.py'): ast.parse(content,filename=name)
    if name.endswith(('.py','.json','.md','.txt')):
        assert b'/Users/alessandromicheli/' not in content,name
        assert b'/data/am1118/' not in content,name
# Resolve public markdown's relative file links. Section anchors are covered by documentation review.
links=[]
for p in root.rglob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' in target or target.startswith('#'): continue
        value=target.split('#',1)[0].strip('<>')
        if not value: continue
        assert (p.parent/value).exists(),(str(p.relative_to(root)),target)
        links.append([str(p.relative_to(root)),target])
qa=base/'export-qa';qa.mkdir()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(base/'mplconfig'),XDG_CACHE_HOME=str(base/'cache'))
env.pop('PYTHONPATH',None)
checks=[]
def execute(name,command):
    result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
    (qa/(name+'.stdout')).write_text(result.stdout)
    (qa/(name+'.stderr')).write_text(result.stderr)
    assert result.returncode==0,(name,result.stderr[-3000:])
    checks.append({'name':name,'command':command,'exit_code':result.returncode})
    return result.stdout
for module in ('run','plot','transport','embeddings','continental'):
    execute('help-'+module,[sys.executable,'-m','paper.'+module,'--help'])
for suite,count in [('tables',30),('liegroups',14),('sweep',126),('highd',12),('ablations',84)]:
    output=qa/('never-created-'+suite)
    result=json.loads(execute('dry-'+suite,[sys.executable,'-m','paper.run',suite,'--gpus','0','--output',str(output),'--dry-run']))
    assert len(result['jobs'])==count and not output.exists(),suite
    assert Path(result['source_root'])==root,suite
execute('dry-continental',[sys.executable,'-m','paper.continental','--config','paper/configs/continental.json','--output',str(qa/'never-created-continental'),'--dry-run'])
plot_python='/Users/alessandromicheli/miniforge3/envs/entropic-rnot/bin/python'
for mode in ('tables','liegroups','sweep','highd','ablations','embeddings'):
    output=qa/'figures'/('reference-'+mode)
    execute('plot-'+mode,[plot_python,'-m','paper.plot',mode,'--reference','--output',str(output)])
    for suffix in ('png','pdf','csv','json'):
        assert (output/(mode+'.'+suffix)).stat().st_size>0,(mode,suffix)
summary={'zip':str(zip_path),'zip_sha256':hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    'files':len(actual),'native_python_files':len(list((root/'src').glob('*.py')))+len(list((root/'rcpm').glob('*.py')))+len(list((root/'experiments').glob('*.py'))),
    'parsed_python_modules':len(list(root.rglob('*.py'))),'resolved_relative_markdown_links':len(links),'checks':checks,
    'scope':'Standalone export checksum/AST/CLI/plan/offline plotting checks. No model training in the local plotting environment.'}
(qa/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks'},indent=2))
