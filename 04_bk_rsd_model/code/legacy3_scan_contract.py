"""模块大纲：记录本次运行配置，只核对数组维度；不是永久的科学政策硬门。

用户可以修改参数、先验、协方差盒数或扫描范围；新运行按新配置记录。
pre/post配置差异仅供报告说明，不因此拒绝用户授权的修改。
已开始的链不能混入另一似然，修改模型时由执行入口另开新链保留旧结果。
"""
import numpy as np


def validate(meta, arrays):
    """只报实际输入的数值/维度错误；不把某次科学选择写死。"""
    names = meta['parameters']
    bounds = np.asarray(arrays['bounds'])
    if bounds.shape != (len(names),2) or np.any(bounds[:,0] >= bounds[:,1]):
        raise ValueError('parameter and prior array dimensions disagree, or lower bound >= upper bound')
    n = meta['P_bins']+meta['B_bins']
    if (arrays['target'].shape != (n,) or arrays['covariance'].shape != (n,n)
            or arrays['chol'].shape != (n,n)):
        raise ValueError('data and covariance dimensions disagree with configured bin counts')
    nc = meta['n_covariance']
    return meta.get('Hartlap_factor', (nc-n-2)/(nc-1))


def configuration_snapshot(meta, arrays):
    """配置可改变；每份结果留下自己的实际设置，不额外设政策门。"""
    validate(meta, arrays)
    fields = ('parameters','stage','kmax_P','kmax_B','P_bins','B_bins','n_target',
        'n_covariance','IR','volume_Gpc_h3','covariance_is_of_mean',
        'covariance_multiplier','P_B_cross_covariance','p_uncertainty_propagated',
        'p_fixed_physical','prior_policy','legacy_noise_closure')
    return {**{key:meta.get(key) for key in fields},
        'bounds':np.asarray(arrays['bounds']).tolist()}


def comparison_differences(first, second, fields=('parameters','bounds','prior_policy','legacy_noise_closure')):
    """差异是说明项，不是拒绝运行的理由。"""
    return {key:{'first':first.get(key),'second':second.get(key)}
        for key in fields if first.get(key) != second.get(key)}
