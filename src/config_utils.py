"""Small helpers for loading JSON configs.

`expand_env` lets a config refer to secrets or machine-specific values as
`${NAME}` or `${NAME:-default}` instead of writing them into the file, so the config
itself can be committed safely.
"""

import os
import re

def load_env_file(path=".env"):
    """Loads KEY=VALUE lines from a .env-style file into os.environ.

    Variables already set in the environment win, so an explicit `export` overrides
    the file. Empty values are skipped. Returns the NAMES that were loaded (never the
    values), so it is safe to print.
    """
    loaded = []
    if not os.path.isfile(path):
        return loaded
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if name and value and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded


_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value, _missing=None):
    """Returns a copy of `value` with ${VAR} / ${VAR:-default} filled from the environment.

    Only strings are touched (numbers, booleans, etc. stay as they are). If a variable
    without a default is not set, one error names every missing variable at once.
    """
    top_level = _missing is None
    missing = [] if top_level else _missing

    if isinstance(value, str):
        def replace(match):
            name, default = match.group(1), match.group(2)
            from_env = os.environ.get(name)
            if from_env:
                return from_env
            if default is not None:
                return default
            missing.append(name)
            return match.group(0)

        result = _PLACEHOLDER.sub(replace, value)
    elif isinstance(value, dict):
        result = {k: expand_env(v, missing) for k, v in value.items()}
    elif isinstance(value, list):
        result = [expand_env(v, missing) for v in value]
    else:
        result = value

    if top_level and missing:
        raise EnvironmentError(
            "Missing environment variable(s) referenced by the config: "
            + ", ".join(sorted(set(missing)))
        )
    return result
