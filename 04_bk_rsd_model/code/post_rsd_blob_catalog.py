"""模块大纲：绕开异常stat/fstat的只读输入；同字节本地暂存；复用原FoF/RSD reader。

不改变质量选择、坐标、速度转换、STO权重或原文件。只在单进程中临时重定向
原reader的输入目录；可选缓存仅复制实际存在的原字节，绝不伪造缺失质量缓存。
"""
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import post_rsd_catalog as adapter

_LOCK = Lock()


def read_blob(path, *, max_bytes=512 * 1024 * 1024):
    """源端只调用open/read/close；不先调用exists/stat/fstat或BufferedReader。"""
    if not isinstance(max_bytes, int) or max_bytes < 0: raise ValueError('nonnegative byte limit required')
    fd = os.open(os.fspath(path), os.O_RDONLY)
    blocks = []; size = 0
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block: break
            size += len(block)
            if size > max_bytes: raise ValueError('input exceeds the declared read limit: ' + str(path))
            blocks.append(block)
    finally:
        os.close(fd)
    return b''.join(blocks)


def optional_blob(path, *, max_bytes=512 * 1024 * 1024):
    try:
        return read_blob(path, max_bytes=max_bytes)
    except FileNotFoundError:
        return None


def catalog_from_blobs(cosmology, realization, raw_blob, *, raw_source,
                       position_cache_blob=None, mass_cache_blob=None):
    """把真实原字节交给原catalog函数；临时路径显式保留在来源元数据中。"""
    if cosmology not in ('fiducial', 'LC_p', 'LC_m') or not isinstance(realization, int) or realization < 0:
        raise ValueError('known cosmology and integer realization required')
    if not isinstance(raw_blob, bytes): raise TypeError('original raw bytes required')
    original = adapter.original
    expected = hashlib.sha256(raw_blob).hexdigest()
    with _LOCK, TemporaryDirectory(prefix='sto_rsd_blob_', dir='/tmp') as scratch:
        temporary = Path(scratch)
        raw = temporary / 'raw' / cosmology / f'real{realization:03d}' / 'groups_003/group_tab_003.0'
        raw.parent.mkdir(parents=True); raw.write_bytes(raw_blob)
        cache = temporary / 'cache' / cosmology / f'real{realization:03d}'
        for name, blob in (('positions_mpc_h.npy', position_cache_blob), ('masses_msun_h.npy', mass_cache_blob)):
            if blob is not None:
                if not isinstance(blob, bytes): raise TypeError('actual optional cache bytes required')
                cache.mkdir(parents=True, exist_ok=True); (cache / name).write_bytes(blob)
        previous_raw, previous_cache = original.RAW_ROOT, original.OLD_POS
        try:
            original.RAW_ROOT = temporary / 'raw'; original.OLD_POS = temporary / 'cache'
            rsd, positions, metadata = adapter.catalog(cosmology, realization)
        finally:
            original.RAW_ROOT, original.OLD_POS = previous_raw, previous_cache
        if metadata['source_raw_sha256'] != expected: raise ValueError('temporary reader input differs from source bytes')
        metadata.update(reader_temporary_source_path=metadata['source_raw'], source_raw=str(raw_source),
            source_read_strategy='os.open/os.read; identical bytes through unchanged catalog reader in local temporary storage',
            original_raw_payload_sha256=expected, original_input_modified=False,
            original_mass_selection_and_RSD_reader_unchanged=True,
            optional_position_cache_provided=position_cache_blob is not None,
            optional_mass_cache_provided=mass_cache_blob is not None)
        return rsd, positions, metadata
