# GPT Pro 审阅任务

请审阅这个 reconstruction/local-PNG 研究快照。先读 START_HERE.md 与 CURRENT_RESULTS.md，再按它们的入口检查实际函数、公式、contract 和结果 JSON。
重点是识别可验证的实现问题和科学解释问题，不把既有行为反过来作为理论定义，也不把 MCMC 收敛视为科学认证。

请逐项回答：

1. fixed catalog 的 fNL_rec/b_rec/p_rec 是否与似然 fNL/b1/p_halo 完全分离？Part3 历史 f_rec=f 路径与 Part4 fixed ABI 的差异是否被正确解释？
2. native Z1/Z2、primordial 外腿、reconstruction shift、G/L/Q 截断与 bphi2 独立算子是否一致？重建分母使用 alpha_rec 与物理 1/M 的区别是否被保留？
3. marked-STO 的 w、w²、w³ 场、自配对扣除、有限 random 与 data-only shift 源是否正确区分？connected 三点/四点响应和 pair closure 还缺哪些可计入当前阶数的项？
4. 4+3/4+4 的实际 legacy stochastic 形状是否足以作为现象诊断？特别检查 PNG 线性项系数、各向异性 R 的 B000 平均与信息量，以及独立 Pshot 是否只进入明确通道。
5. p 校准采用 weighted input、真正 post P0、实空间 p、联合 MAP/边缘中位数的各次对照是否自洽？校准/目标与目标/covariance 重叠、p 不确定度未传播怎样限制恢复解释？
6. 两腿 cutoff、unrestricted closing leg、实际离散窗口、CIC/TSC/interlacing 与 shot helper 是否匹配？是否把 strict-three-leg B0 与 Sugiyama B000 混用了？
7. real-space/RSD 的 b2/gamma2/bK2 转换和 bphidelta 的 PBS anchor 是否一致？IR single-wiggle undamped 源与 hard tree 的 matching remainder 是否被错误算作阻尼效应？
8. 哪些较高 k 偏移更可能来自 Gaussian baseline、PNG response、噪声闭合、边缘化或先验？现有证据能排除哪些解释，仍需什么受控对照？

输出格式：先列最重要问题；每项给出文件/函数或行号、公式/数值证据、影响范围、可信度以及最小验证方案。
分别标记“实现错误”“模型尚未闭合”“统计解释限制”“已有证据通过”。对尚无证据的归因明确写假设。
不要要求先递归读取全部 outputs/ 或历史链。当前快照刻意只提供精选结果，完整文件来源见 REVIEW_MANIFEST.json。
