"""模块大纲：只替换原并行case的checkpoint预算；其他range调用不变。"""
import builtins


def checkpoint_range(previous_steps, maximum_steps):
    if (isinstance(previous_steps, bool) or isinstance(maximum_steps, bool)
            or previous_steps < 6000 or previous_steps % 2000
            or maximum_steps <= previous_steps or maximum_steps % 2000):
        raise ValueError('equal original checkpoint and explicit larger budget required')

    def routed(*args):
        if args == (6000, 64001, 2000):
            return builtins.range(previous_steps + 2000, maximum_steps + 1, 2000)
        return builtins.range(*args)

    return routed
