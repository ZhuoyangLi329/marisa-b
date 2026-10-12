"""模块大纲：复用完整校准核验，但将新审计写到独立快照，不改已有冻结报告。

旧verify会刷新COMPLETION_AUDIT时间；冻结P0已引用其SHA，不可被重审覆盖。
只路由输出，原全HDF/数值/先验/RNG检查不改；p取原校准中位数。
"""
from pathlib import Path
import audit_all_available_rsd_pcal_v1 as original
from rsd_b000_contract import digest

def recheck(stage,destination):
    root=original.run.stage_root(stage);case=root/'cases'/original.run.case_name(stage)
    protected=[root/'COMPLETION_AUDIT.json',case/'SUMMARY.json',case/'SUMMARY_METADATA_CORRECTED.json']
    before={str(p):digest(p) for p in protected}
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    save_original=original.save_json
    def save(path,value):
        name=Path(path).name
        if name not in ('COMPLETION_AUDIT.json','SUMMARY_METADATA_CORRECTED.json'):
            raise ValueError('unexpected calibration write during read-only recheck')
        return save_original(destination/name,value)
    original.save_json=save
    try:p,report=original.verify(stage)
    finally:original.save_json=save_original
    after={str(p):digest(p) for p in protected}
    if after!=before:raise ValueError('read-only calibration recheck modified old frozen evidence')
    save_original(destination/'FROZEN_EVIDENCE_UNCHANGED.json',dict(stage=stage,
        original_protected_sha256=before,old_audit_summary_and_corrected_metadata_preserved=True,
        actual_chain_sha256=digest(case/'chain.h5'),physical_p=p,
        original_audit_source_sha256=digest(original.__file__),adapter_sha256=digest(__file__)))
    return p,report
