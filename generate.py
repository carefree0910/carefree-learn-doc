import os
import json

from typing import Any
from typing import Set
from typing import Dict


def inject_requires(require_keys: Set[Any], local_requires: Dict[str, Any]) -> None:
    for v in local_requires.values():
        if isinstance(v, dict):
            inject_requires(require_keys, v)
            continue
        for vv in v:
            require_keys.add(vv)


def supported_models(configs_folder: str, export_path: str) -> None:
    prefix = "cflearn/api/zoo/configs"
    body = []
    for task in sorted(os.listdir(configs_folder)):
        body.extend([f"### `{task}`\n"])
        task_folder = os.path.join(configs_folder, task)
        for model in sorted(os.listdir(task_folder)):
            body.append(f"#### `{model}`")
            model_folder = os.path.join(task_folder, model)
            for json_file in sorted(os.listdir(model_folder)):
                mtype = os.path.splitext(json_file)[0]
                body.append(f"##### {mtype}")
                with open(os.path.join(model_folder, json_file), "r") as f:
                    config = json.load(f)
                require_keys = set()
                inject_requires(require_keys, config.get("__requires__", {}))
                if not require_keys:
                    require_msg = ""
                else:
                    require_msg = ", ".join([f"{k}=..." for k in sorted(require_keys)])
                    require_msg = f", {require_msg}"
                appendix = "" if mtype == "default" else f".{mtype}"
                body.extend(
                    [
                        "```python",
                        f"# {prefix}/{task}/{model}/{json_file}",
                        f'm = load("{task}/{model}{appendix}"{require_msg})',
                        "```"
                    ]
                )
            body.append("")
    with open(export_path, "w") as f:
        f.write("\n".join(body))
