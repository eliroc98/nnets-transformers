"""Fetch a checksummed official supplement; generated/downloaded data stay ignored."""
import hashlib
import importlib.util
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://proceedings.neurips.cc/paper_files/paper/2024/file/47c7edadfee365b394b2a3bd416048da-Supplemental-Conference.zip'
SHA256 = '89630b437cc49501597b9fcb9b2f994c7b52a97369e0153f4280c5cd107096c1'


def source(root):
    root = Path(root)
    archive = root / 'supplement.zip'
    if not archive.exists():
        root.mkdir(parents=True, exist_ok=True)
        temporary = root/'supplement.zip.download'
        urllib.request.urlretrieve(URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Official supplement checksum mismatch')
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Official supplement checksum mismatch')
    # Verify every extracted source against the immutable archive on every use.
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith('circuits-over-time/') and name.endswith(('.py', '.csv')) and (
                '/data/' in name or name.endswith('/path_patching_cm/ioi_dataset.py')):
                relative = Path(name).relative_to('circuits-over-time')
                if '..' in relative.parts:
                    raise ValueError('Unsafe supplement member')
                target = root/relative
                expected = bundle.read(name)
                if target.exists() and target.read_bytes() != expected:
                    raise ValueError(f'Supplement source changed: {relative}')
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(expected)
    return root


def module(path):
    spec = importlib.util.spec_from_file_location('verified_tigges_'+Path(path).stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
