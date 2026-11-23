"""Top-level worker entry (picklable under spawn)."""


def work(task):
    exp = task["exp"]
    if exp == "X8":
        from .x8 import cell
        return cell(task)
    if exp == "X9A":
        from .x9a import instance_task
        return instance_task(task)
    if exp == "X9B":
        from .x9b import task_fn
        return task_fn(task)
    if exp == "X10":
        from .x10 import x10a_task, x10b_task
        return x10a_task(task) if task["part"] == "a" else x10b_task(task)
    if exp == "X1E":
        from .x1e import lowoverlap_task, saturation_task
        return lowoverlap_task(task) if task["part"] == "low" else saturation_task(task)
    if exp == "X11":
        from .x11 import x11_task
        return x11_task(task)
    raise ValueError(exp)
