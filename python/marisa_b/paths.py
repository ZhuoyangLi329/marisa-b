"""不依赖历史 worktree 布局的仓库与数据路径解析。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DATA_ROOT_ENVIRONMENT_VARIABLE = "MARISA_B_DATA_ROOT"


def _absolute_path(path: Path, *, anchor: Path | None = None) -> Path:
    """展开用户目录，并把相对路径锚定到指定目录后规范化。"""

    expanded = path.expanduser()
    if not expanded.is_absolute() and anchor is not None:
        expanded = anchor / expanded
    return expanded.resolve()


@dataclass(frozen=True)
class RepositoryPaths:
    """MARISA-B 源码根目录和外部科学数据根目录。"""

    repo_root: Path
    data_root: Path

    def data_path(self, *parts: str | Path) -> Path:
        """解析数据根目录下的相对路径并拒绝目录逃逸。

        Parameters
        ----------
        parts
            数据根目录下的一个或多个相对路径片段。

        Returns
        -------
        pathlib.Path
            规范化后的绝对数据路径。
        """

        if not parts:
            return self.data_root
        candidate = self.data_root.joinpath(*parts).resolve()
        if not candidate.is_relative_to(self.data_root):
            raise ValueError("data path escapes MARISA_B_DATA_ROOT")
        return candidate


def resolve_repository_paths(
    *,
    repo_root: str | Path | None = None,
    data_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> RepositoryPaths:
    """从唯一允许的三类输入解析仓库和数据根目录。

    Parameters
    ----------
    repo_root
        显式仓库根目录；省略时由 ``python/marisa_b`` 的安装源码位置推断。
    data_root
        CLI ``--data-root`` 对应值，优先级高于环境变量。
    environment
        用于测试或调用方注入的环境映射；默认读取 ``os.environ``。

    Returns
    -------
    RepositoryPaths
        规范化后的仓库和数据根目录。

    Notes
    -----
    优先级为显式 ``data_root``、``MARISA_B_DATA_ROOT``、``repo_root``。
    相对数据根目录以仓库根目录为锚点。这里没有任何嵌套 worktree 或
    ``history`` 目录假设。
    """

    if repo_root is None:
        resolved_repo = Path(__file__).resolve().parents[2]
    else:
        resolved_repo = _absolute_path(Path(repo_root))
    source_environment = os.environ if environment is None else environment
    raw_data_root: str | Path
    if data_root is not None:
        raw_data_root = data_root
    elif source_environment.get(DATA_ROOT_ENVIRONMENT_VARIABLE):
        raw_data_root = source_environment[
            DATA_ROOT_ENVIRONMENT_VARIABLE
        ]
    else:
        raw_data_root = resolved_repo
    resolved_data = _absolute_path(
        Path(raw_data_root),
        anchor=resolved_repo,
    )
    return RepositoryPaths(
        repo_root=resolved_repo,
        data_root=resolved_data,
    )
