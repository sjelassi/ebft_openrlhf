import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _json_safe(x: Any) -> Any:
    """Best-effort conversion to something JSON serializable."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    # Fallback (e.g. Path, numpy scalars, enums, etc.)
    return str(x)


def _prune_client_states(client_states: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Avoid dumping huge / non-serializable client states into the run config."""
    if not client_states:
        return client_states
    pruned = dict(client_states)
    # This can be large and is primarily used for resuming; omit from human-facing run config.
    if "data_loader_state_dict" in pruned:
        pruned["data_loader_state_dict"] = "<omitted>"
    return pruned


def write_run_config(
    output_dir: str,
    args: Any,
    *,
    tag: Optional[str] = None,
    client_states: Optional[Dict[str, Any]] = None,
    filename: str = "run_config.json",
) -> None:
    """
    Write a small JSON snapshot of the run configuration into a checkpoint directory.

    Includes:
      - args (argparse.Namespace -> vars(args) best-effort)
      - client_states (pruned)
      - metadata (timestamp, hostname, cwd, tag)
    """
    os.makedirs(output_dir, exist_ok=True)

    args_dict = vars(args) if hasattr(args, "__dict__") else args
    payload = {
        "tag": tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "args": _json_safe(args_dict),
        "client_states": _json_safe(_prune_client_states(client_states)),
    }

    tmp_path = os.path.join(output_dir, filename + ".tmp")
    final_path = os.path.join(output_dir, filename)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, final_path)


