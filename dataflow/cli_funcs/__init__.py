def cli_init(*args, **kwargs):
    from .cli_init import cli_init as _cli_init
    return _cli_init(*args, **kwargs)


def cli_env(*args, **kwargs):
    from .cli_env import cli_env as _cli_env
    return _cli_env(*args, **kwargs)


def cli_json2kg_init(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_init as _cli_json2kg_init
    return _cli_json2kg_init(*args, **kwargs)


def cli_json2kg_autoschemakg(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_autoschemakg as _cli_json2kg_autoschemakg
    return _cli_json2kg_autoschemakg(*args, **kwargs)


def cli_json2kg_wikontic(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_wikontic as _cli_json2kg_wikontic
    return _cli_json2kg_wikontic(*args, **kwargs)


def cli_json2kg_kggen(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_kggen as _cli_json2kg_kggen
    return _cli_json2kg_kggen(*args, **kwargs)


def cli_json2kg_treekg(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_treekg as _cli_json2kg_treekg
    return _cli_json2kg_treekg(*args, **kwargs)


def cli_json2kg_kgflow(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_kgflow as _cli_json2kg_kgflow
    return _cli_json2kg_kgflow(*args, **kwargs)


def cli_json2kg_eval(*args, **kwargs):
    from .cli_json2kg import cli_json2kg_eval as _cli_json2kg_eval
    return _cli_json2kg_eval(*args, **kwargs)


__all__ = [
    "cli_env",
    "cli_init",
    "cli_json2kg_init",
    "cli_json2kg_autoschemakg",
    "cli_json2kg_wikontic",
    "cli_json2kg_kggen",
    "cli_json2kg_treekg",
    "cli_json2kg_kgflow",
    "cli_json2kg_eval",
]
