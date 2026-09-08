#!/usr/bin/env python3
"""Build a self-contained paper-code ZIP, excluding the research archive."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

FILES=('README.md','LICENSE','CITATION.cff','requirements.txt','.gitignore')
DIRECTORIES=('src','rcpm','experiments','paper','requirements','reference','data','docs','tools')
REQUIRED=('src/manifolds.py','rcpm/LICENSE','paper/run.py','paper/worker.py','paper/plot.py',
          'paper/transport.py','paper/embeddings.py','paper/continental.py',
          'requirements/cuda.txt','reference/paper_targets.json','docs/experiments.md')

def source_files(root):
    for name in (*FILES,*REQUIRED):
        if not (root/name).is_file(): raise ValueError(f'Missing release file: {name}')
    selected=[root/name for name in FILES]
    for directory in DIRECTORIES:
        for path in (root/directory).rglob('*'):
            if path.is_symlink(): raise ValueError(f'Release input may not be a symlink: {path}')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.pyo') and path.name!='.DS_Store':
                selected.append(path)
    return sorted(set(selected),key=lambda p:p.relative_to(root).as_posix())

def build(root,output):
    root=root.resolve(); output=output.resolve()
    if output.exists(): raise FileExistsError(f'Choose a new output path: {output}')
    selected=source_files(root)
    if output in selected or any(output.is_relative_to(root/name) for name in DIRECTORIES):
        raise ValueError('Write the ZIP outside release source directories, for example under dist/.')
    data={p.relative_to(root).as_posix():p.read_bytes() for p in selected}
    manifest={'schema_version':1,'project':'Riemannian Neural Optimal Transport',
        'paper':'https://openreview.net/forum?id=ez4oLq7PR3',
        'scope':'Paper experiment code, plotting, data and numerical references. Historical audits and website excluded.',
        'files':{name:{'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value)} for name,value in data.items()}}
    data['RELEASE_MANIFEST.json']=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            for name,value in sorted(data.items()):
                info=zipfile.ZipInfo('riemannian-neural-optimal-transport/'+name,date_time=(1980,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                info.external_attr=0o100644<<16
                archive.writestr(info,value,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    return {'file':str(output),'files':len(data),'bytes':output.stat().st_size,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try: result=build(args.source_root,args.output)
    except (ValueError,FileExistsError) as error: parser.error(str(error))
    print(json.dumps(result,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
