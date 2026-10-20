import importlib


def import_optional(module: str, extra: str):
    """
    Imports a module that belongs to one of sparks-ai's optional dependencies.

    Args:
        module (str): Dotted path of the module to import, e.g. 'cebra.datasets'.
        extra (str): The sparks-ai pip extra that installs it (see pyproject.toml), named in the
            error message if the import fails.

    Returns:
        The imported module.

    Raises:
        ImportError: If the module can't be imported, telling the user which extra to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        package = module.split('.')[0]
        raise ImportError(f"This feature requires `{package}`, which is not installed. "
                          f"Install it with `pip install sparks-ai[{extra}]`.") from e
