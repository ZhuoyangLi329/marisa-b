#!/usr/bin/env python3
"""模块大纲：核对交付 oracle、测试日志、模板身份；记录改动/备份；输出验收边界。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/"marisa_b_core"
OUT=ROOT/"halo/outputs/local_png_post_tree_ir_20260930"


def record(path):
    path=Path(path)
    return {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}


def main():
    audit=json.loads((OUT/"audit_fine/POST_TREE_IR_AUDIT.json").read_text())
    assert audit["status"].startswith("PASS_selected_nodes"),"numerical audit incomplete"
    tests=(OUT/"tests.log").read_text();build=(OUT/"build.log").read_text()
    passed=int(re.search(r"(\d+) passed",tests).group(1))
    checks=int(re.search(r"checks passed: (\d+)",build).group(1))
    assert "failed" not in tests and "ERROR" not in tests
    imports={
        "python/marisa_b/post_recon_tree_png_ir.py":"fe9cdd352abde8029c1422915212017148852c295ae349818d39acbc121e9b1c",
        "tests/python/test_post_recon_tree_png_ir.py":"647cbc249dcc436b1de1f6355b7c038370a771f5b0ae98ed1a3ec2d1cd9c2571",
        "docs/POST_RECON_TREE_PNG_IR_API.md":"8c99fb6faeceb8c7a824dc634b360a1b4675ada3c9be50dcae459e7af6dd83cb",
        "docs/POST_RECON_TREE_PNG_IR_RESUMMATION_PLAN.md":"d5266602f4d6abc5958967e512859cc60907dd4a353b60ead40adc5ddc3cd4ec",
    }
    for name,expected in imports.items():
        assert record(CORE/name)["sha256"]==expected, f"import changed: {name}"
    assert record(CORE/"python/marisa_b/tree_png_ir_resummation.py")["sha256"]=="762756a7e0da1568a2395ba17d383c2b2268325d1a1bd5399eb19393e8718af5"
    sources=["README.md","src/marisa_b/tree_png_ir.h","src/marisa_b/tree_png_ir.cpp",
        "src/marisa_b/post_recon_tree_png_ir.h","src/marisa_b/post_recon_tree_png_ir.cpp",
        "scripts/build_tree_png_ir.sh","python/marisa_b/post_recon_tree_png_ir_native.py",
        "tests/eft_v2/test_tree_png_ir_native.cpp","tests/python/test_post_recon_tree_png_ir_native.py",
        "scripts/production/generate_post_tree_png_ir.py","scripts/examples/reconstruction_current_p.py",
        "configs/reconstruction_current_p_example.json","docs/POST_RECON_TREE_PNG_IR_NATIVE.md",*imports]
    identities={}
    for p in sorted((OUT/"audit_fine").glob("*_shell_templates.json")):
        metadata=json.loads(p.read_text())
        assert record(metadata["output"]["path"])["sha256"]==metadata["output"]["sha256"]
        for key,input_record in metadata["inputs"].items():
            assert record(input_record["path"])["sha256"]==input_record["sha256"],f"template identity drift: {key}"
        identities[p.name]=metadata
    assert len(identities)==3
    names={"standard_gra":"std","marisa_gra":"Marisa","sto_marisa":"STO+Marisa"}
    angular=max(x["G_L_Q"] for row in audit["methods"].values() for x in row["angular_scaled_errors"].values())
    radial=max(row["radial_scaled_errors"]["2049"]["G_L_Q"] for row in audit["methods"].values())
    variance=max(row["radial_scaled_errors"]["2049"]["variance"] for row in audit["methods"].values())
    shell=max(row["shell"]["angular_scaled_error"] for row in audit["methods"].values())
    summary={"implementation":"native_post_tree_local_png_LO_IR_integrated_opt_in",
             "scientific_status":"provisional; post_catalogue_and_full_kmax_likelihood_not_validated",
             "native_source_checks":checks,"targeted_tests_passed":passed,
             "all_python_tests":"107 passed, 6 failed, 14 errors; failures are missing nonlocal/production files",
             "release_test":"blocked at repository boundary: migrated NERSC directory has no .git metadata",
             "methods":list(names),"parameter_path":"f_rec=eta*f only",
             "angular_G_L_Q_scaled_error_max":angular,"radial_1025_2049_G_L_Q_scaled_error_max":radial,
             "radial_1025_2049_variance_scaled_error_max":variance,"first_shell_angular_scaled_error_max":shell,
             "physical_limits":["real space", "local PNG", "O(Pw)", "O(fNL^2)", "Gaussian linear displacement"],
             "not_done":["fixed nonzero f_rec catalogue expansion", "dedicated post bias/noise calibration",
                         "all-shell covariance-weighted convergence", "trusted post PNG injection recovery", "B-only kmax scan"]}
    (OUT/"RESULTS_SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
    lines=["# Post-reconstruction local-PNG tree IR：实现与验证报告", "",
        "已将交付方案接入 NERSC MARISA-B C++ tree 内核，并提供 Python/批量 shell 模板显式入口。",
        "适用于 std、Marisa 和 STO+Marisa 的方案参数路径；旧 likelihood 默认值与历史结果未改动。", "",
        "## 本次补齐", "",
        "- 固定/自适应位移的 Sigma(fNL) 到二阶；包括 std 分子的 scale-dependent bias 响应。",
        "- 三个 adaptive 源基底、严格单 wiggle、G→L / L→Q / G→Q 指数混合。",
        "- Cartesian CIC 逐外腿方向积分，以及 IR 后按原分母做 shell 投影。",
        "- 可选 shift noise；stochastic 源项与 mixing 分离；不默认强加公共 alpha3 closure。", "",
        "## 验证结果", "",
        f"- 当前 native 基底交叉检查：{checks:,} 项通过；独立/回归 Python 测试：{passed} 个通过。",
        f"- 真实 z=0.5 谱三方法探针，24x48/32x64/48x96 角积分 G/L/Q 最大 scaled error：{angular:.3e}。",
        f"- 1025→2049 径向网格：G/L/Q 最大 {radial:.3e}；位移方差最大 {variance:.3e}。",
        f"- 首个真实 FFT shell 306 节点、原分母 324；角积分变化最大 {shell:.3e}；分块模板与直接投影一致。",
        "- qmin/kIR/rbao 敏感性、fNL=±100 方差截断检查已记录在 audit_fine/POST_TREE_IR_AUDIT.json。",
        "- scaled error = max(abs(a-b)/max(1,abs(b)))，不是观测协方差加权误差。", "",
        "## 不能混淆的边界", "",
        "- fNL_rec=eta*fNL：std eta=0，adaptive 示例 eta=1；固定非零 fNL_rec 目录需要另一个展开。",
        "- current-p 示例使用 parent pre/STO 校准；不是新 post 校准，演示 b2/gamma2 也不是拟合结果。",
        "- 本次没有重跑可信 post PNG 目录的 B-only kmax 扫描，不能据此断言 kmax 提升。",
        "- make release-test 因工作副本没有 Git 元数据未通过。全仓库 Python 测试为 107 通过、6 失败、14 错误；20 个未通过项均引用未迁入的 nonlocal/production 文件。",
        "- 已有 post provisional 与 halo PNG catalogue 限制保持不变。", "",
        "## 入口与产物", "",
        "- 使用说明：marisa_b_core/docs/POST_RECON_TREE_PNG_IR_NATIVE.md。",
        "- 完整模型：marisa_b_core/python/marisa_b/post_recon_tree_png_ir_native.py。",
        "- 批量投影：marisa_b_core/scripts/production/generate_post_tree_png_ir.py。",
        "- 实例：audit_fine/ 下三方法的 *_config.json、*_exact_shell_nodes.npz、*_shell_templates.npz。",
        "- 图：audit_fine/post_tree_ir_current_p.pdf；没有生成 PNG。",
        "- 原文件备份：original_source/；源码/输入/输出哈希：POST_TREE_IR_CHANGE_MANIFEST.json。", ""]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    assert not list(OUT.rglob("*.png")),"unexpected PNG artifact"
    destination=OUT/"POST_TREE_IR_CHANGE_MANIFEST.json"
    payload={"schema":"marisa-post-tree-ir-change-manifest-v1","created_utc":datetime.now(timezone.utc).isoformat(),
             "task_repository":str(ROOT.parent),"user_package":"marisa-b-post-recon-tree-png-ir, 2026-09-30",
             "imported_file_sha256":imports,"source_files":[record(CORE/n) for n in sources]+
                 [record(ROOT/"halo/code"/n) for n in ("audit_post_tree_png_ir.py","write_post_tree_ir_manifest.py")],
             "current_p_example_origin":"ZhuoyangLi329/marisa-b PR 3 commit 615192fe4cedabe5e75148686418c1a9a6b7bcf4",
             "unchanged_native_reference":[record(CORE/"src/marisa_b"/n) for n in ("marisa_b_native.cpp","marisa_b_native.h")],
             "backups":[record(p) for p in sorted((OUT/"original_source").rglob("*")) if p.is_file()],
             "template_identities":identities,"summary":summary,
             "output_files":[record(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p!=destination and "original_source" not in p.parts],
             "historical_outputs_modified":False,"historical_pre_manifest_rewritten":False,
             "github_published":False}
    destination.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
