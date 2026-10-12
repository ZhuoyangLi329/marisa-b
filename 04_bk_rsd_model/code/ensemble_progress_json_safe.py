"""模块大纲：只将writer的numpy整数步数转成JSON可保存的Python int。"""
from independent_ensemble_backend import advance_ensemble


def advance_json_safe(case,filename,ensemble,case_index,steps,progress=None):
    safe=None if progress is None else lambda actual:progress(int(actual))
    return advance_ensemble(case,filename,ensemble,case_index,steps,safe)
