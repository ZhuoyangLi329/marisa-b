#!/usr/bin/env python3
"""模块大纲：显式输入实际方向节点和谱；分块 native IR；保留原分母投影；内容哈希。

输入 NPZ: vectors(N,3,3), power/no_wiggle/transfer(N,3), q/qpower/qnw/qm(Q),
weights(N), bin_ids(N); 可选 shift_noise(Q)。JSON 配置不能省略方向和分母声明。
这里不自动选目录、不拟合参数、不复用 pre 投影节点、不合并 stochastic closure。
"""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from marisa_b.tree_png_ir_resummation import HaloTreeBias, ReconstructionConfig
from marisa_b.post_recon_tree_png_ir import AdaptiveReconstructionConfig, PostReconIrSettings
from marisa_b.post_recon_tree_png_ir_native import NativePostTreeIR, POST_NATIVE_TERMS, project_post_tree_nodes


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def generate(input_path, config_path, output_path, library, chunk_size=128):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("output must end in .npz")
    manifest_path = output_path.with_suffix(".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing templates or manifest")
    config = json.loads(Path(config_path).read_text())
    if config["reconstruction_parameter_path"] != "f_rec=eta*f":
        raise ValueError("only f_rec=eta*f is implemented; fixed nonzero f_rec needs another expansion")
    if config["weights_include_original_denominator"] is not True:
        raise ValueError("original denominator is required")
    bias = HaloTreeBias(**config["bias"])
    rec = ReconstructionConfig(**config["reconstruction"])
    adaptive = AdaptiveReconstructionConfig(**config["adaptive"])
    settings = PostReconIrSettings(**config["ir"])
    if rec.cell_size > 0 and rec.cic_window_power > 0 and config.get("absolute_orientation_nodes") is not True:
        raise ValueError("active CIC requires explicit absolute orientation nodes")
    engine = NativePostTreeIR(library)
    sums = None
    with np.load(input_path, allow_pickle=False) as z:
        data = {key: z[key] for key in ("vectors", "weights", "bin_ids", "power", "no_wiggle", "transfer")}
        n = len(data["vectors"])
        if not n:
            raise ValueError("empty node input")
        for key in ("weights", "bin_ids", "power", "no_wiggle", "transfer"):
            if len(data[key]) != n:
                raise ValueError(f"{key} does not match node count")
        qgrid = [z[name] for name in ("q", "qpower", "qnw", "qm")]
        noise = z["shift_noise"] if "shift_noise" in z else None
        for start in range(0, n, chunk_size):
            sl = slice(start, min(n, start+chunk_size))
            batch = engine.evaluate(data["vectors"][sl], *(data[key][sl] for key in ("power", "no_wiggle", "transfer")),
                                    bias, rec, *qgrid, settings, adaptive, shift_noise_power=noise)
            part = project_post_tree_nodes(batch,data["weights"][sl],data["bin_ids"][sl],
                                          number_of_bins=config["number_of_bins"],
                                          weights_include_original_denominator=True)
            if sums is None:
                sums = part
            else:
                for key in sums:
                    sums[key] += part[key]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **sums)
    manifest = {"schema": "marisa-post-tree-ir-templates-v1", "status": "numerical_templates_not_catalogue_validation",
                "configuration": config, "inputs": {"nodes": record(input_path), "config": record(config_path),
                "native_library": record(library)}, "output": record(output_path), "number_of_nodes": n,
                "source_columns": POST_NATIVE_TERMS, "deterministic_columns": ["G", "L", "Q"],
                "mixing_columns": ["L_from_G", "Q_from_L", "Q_from_G"],
                "shift_noise_enabled": noise is not None, "stochastic_closure": "not_combined",
                "scope": "real-space local-PNG tree; O(Pw), O(fNL^2), Gaussian linear long displacement"}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "build/tree_png_ir/libmarisa_post_tree_png_ir.so")
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(generate(args.input,args.config,args.output,args.library,args.chunk_size), indent=2))


if __name__ == "__main__":
    main()
