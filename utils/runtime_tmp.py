"""Keep multiprocessing IPC files on local storage, away from NFS .nfs locks."""
import os
from pathlib import Path
import tempfile


def configure_local_temporary_directory():
    """Call before creating workers; preserve an explicit HEARTFFD_TMPDIR.

    TMPDIR/TEMP/TMP can be inherited from an older NAS-backed environment.
    multiprocessing uses tempfile for listener sockets; NFS may retain those
    as busy .nfs files during worker finalization. Checkpoints still use their
    explicit output directories. Never delete existing IPC directories here.
    """
    requested = os.environ.get('HEARTFFD_TMPDIR') or f'/tmp/heartssm-{os.getuid()}'
    target = Path(requested).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Fail early if the chosen local scratch is not writable.
    with tempfile.TemporaryFile(dir=target):
        pass
    for name in ('TMPDIR', 'TEMP', 'TMP', 'HEARTFFD_TMPDIR'):
        os.environ[name] = str(target)
    # tempfile caches its choice on first use, including during imports.
    tempfile.tempdir = None
    return target
