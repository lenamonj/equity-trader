def gathered_for(module) -> dict[str, dict]:
    """What the gatherer would hand this desk, with every section empty."""
    return {section.__name__: {"data": {}, "error": None} for section in module.DATA}
