"""Stdlib-only completeness, path, size and SHA-256 checks; no writes."""
from pathlib import Path
import hashlib
BASE=Path(__file__).resolve().parents[1]
manifest={}
for line in (BASE/'SHA256SUMS').read_text().splitlines():
 digest,rel=line.split('  ',1)
 assert len(digest)==64 and rel not in manifest
 p=Path(rel)
 assert not p.is_absolute() and '..' not in p.parts
 manifest[rel]=digest
files={str(p.relative_to(BASE)):p for p in BASE.rglob('*') if p.is_file()}
assert set(manifest)==set(files)-{'SHA256SUMS'},'manifest completeness'
for p in BASE.rglob('*'):
 assert not p.is_symlink(),'symlink'
 assert 'private' not in p.relative_to(BASE).parts,'excluded directory'
for rel,p in files.items():
 assert p.stat().st_size<=20_000_000,rel
 if rel!='SHA256SUMS':assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[rel],rel
print('PASS: complete manifest,',len(files),'files; sizes, paths, symlinks and SHA-256 verified.')
