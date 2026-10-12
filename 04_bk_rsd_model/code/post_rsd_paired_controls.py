"""模块大纲：同盒数值控制的配对差；区分单盒散布与诊断均值误差。

只返回配对诊断，不构造likelihood covariance，不作逆矩阵或fNL精度认证。
sqrt(N)仅用于估计这批数值控制之平均差的误差，不改变主分析C_single。
"""
import numpy as np


def paired_moments(baseline, control):
    """第一轴必须是相同顺序的realization；来源身份由调用方单独核验。"""
    base = np.asarray(baseline, dtype=float)
    other = np.asarray(control, dtype=float)
    if base.ndim != 2 or base.shape != other.shape or base.shape[0] < 2 or base.shape[1] < 1:
        raise ValueError('same nonempty [realization, bin] shapes and at least two pairs required')
    if not np.isfinite(base).all() or not np.isfinite(other).all():
        raise ValueError('finite paired measurements required')
    delta = other - base
    std = base.std(axis=0, ddof=1)
    if np.any(std <= 0): raise ValueError('positive baseline single-box dispersion required')
    mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0, ddof=1)
    return {'baseline_mean': base.mean(axis=0), 'control_mean': other.mean(axis=0),
            'baseline_single_box_std': std, 'control_single_box_std': other.std(axis=0, ddof=1),
            'paired_delta': delta, 'paired_mean_delta': mean, 'paired_delta_std': delta_std,
            'diagnostic_SE_of_paired_mean': delta_std / np.sqrt(len(base)),
            'paired_mean_over_single_box_std': mean / std,
            'paired_RMS_over_single_box_std': np.sqrt(np.mean(delta * delta, axis=0)) / std}


def selected_summary(moments, indices):
    """所选bin的描述性量；不把多bin最大值当成显著性或完整白化范数。"""
    ix = np.asarray(indices)
    if (ix.ndim != 1 or not np.issubdtype(ix.dtype, np.integer) or len(ix) < 1
            or len(np.unique(ix)) != len(ix) or np.any(ix < 0)
            or np.any(ix >= len(moments['baseline_single_box_std']))):
        raise ValueError('unique valid bin indices required')
    scale = moments['baseline_single_box_std'][ix]
    normalized = moments['paired_mean_delta'][ix] / scale
    se = moments['diagnostic_SE_of_paired_mean'][ix] / scale
    worst = int(np.argmax(np.abs(normalized)))
    return {'bins': len(ix), 'max_abs_mean_delta_over_single_box_std': float(np.max(np.abs(normalized))),
            'RMS_mean_delta_over_single_box_std': float(np.sqrt(np.mean(normalized * normalized))),
            'worst_mean_delta_bin_index': int(ix[worst]),
            'signed_mean_delta_over_std_at_worst_bin': float(normalized[worst]),
            'diagnostic_SE_over_std_at_worst_bin': float(se[worst]),
            'max_diagnostic_SE_over_single_box_std': float(np.max(se)),
            'max_paired_RMS_over_single_box_std': float(np.max(moments['paired_RMS_over_single_box_std'][ix])),
            'covariance_whitened': False, 'fNL_bias_or_convergence_certified': False}
