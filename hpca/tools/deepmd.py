"""
DeepMD-kit tool: write input.json, submission scripts, parse training curves,
test models, prepare datasets from AIMD, freeze/compress.
"""
from __future__ import annotations

import json
import subprocess
from hpca.core.config import account_fallback as _account_fallback
from hpca.core.slurm_submit import module_bundle_lines as _mbl
from pathlib import Path
from typing import Optional

from .base import Tool, ToolResult


def _hpc(key: str, default: str = "") -> str:
    """Read a value from platform.yaml hpc section at call time (lazy, no import-time side effects)."""
    try:
        from hpca.core.paths import load_platform_config
        return load_platform_config().get("hpc", {}).get(key, default)
    except Exception:
        return default


def _dp_bin() -> str:
    """Return the dp binary path from platform.yaml or fall back to venv/bin/dp."""
    explicit = _hpc("deepmd_bin", "")
    if explicit:
        return explicit
    venv = _hpc("deepmd_cpu_venv", "")
    if venv:
        candidate = f"{venv}/bin/dp"
        from pathlib import Path
        if Path(candidate).exists():
            return candidate
    return "dp"  # assume on $PATH


def _dp_env() -> str:
    """Return the DeepMD conda environment path derived from platform.yaml python_deepmd."""
    return _hpc("python_deepmd", "").replace("/bin/python3", "")


def _dp_python() -> str:
    """Return the Python interpreter path for the DeepMD environment from platform.yaml."""
    return _hpc("python_deepmd", "python3")


def __getattr__(name: str) -> str:
    """Lazy module-level resolution so DP_BIN/DP_ENV/DP_PYTHON read platform.yaml at access time."""
    if name == "DP_BIN":
        return _dp_bin()
    if name == "DP_ENV":
        return _dp_env()
    if name == "DP_PYTHON":
        return _dp_python()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_DEEPMD_INPUT_TEMPLATE = {
    "model": {
        "type_map": [],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 0.5,
            "sel": [],
            "neuron": [25, 50, 100],
            "axis_neuron": 16,
            "seed": 42,
        },
        "fitting_net": {
            "neuron": [240, 240, 240],
            "resnet_dt": True,
            "seed": 42,
        },
    },
    "learning_rate": {
        "type": "exp",
        "start_lr": 1e-3,
        "stop_lr": 3.51e-8,
        "decay_steps": 2000,
    },
    "loss": {
        "start_pref_e": 0.02, "limit_pref_e": 1,
        "start_pref_f": 1000, "limit_pref_f": 1,
        "start_pref_v": 0,    "limit_pref_v": 0,
    },
    "training": {
        "stop_batch": 1_000_000,
        "disp_file": "lcurve.out",
        "disp_freq": 1000,
        "numb_test": 10,
        "save_freq": 10000,
        "save_ckpt": "model.ckpt",
        "training_data": {
            "systems": ["00.data/set.000"],
            "batch_size": 32,
        },
        "validation_data": {
            "systems": ["00.data/validation_data"],
            "batch_size": 32,
        },
    },
}


class DeepMDTool(Tool):
    """Tool wrapper for DeepMD-kit: writes configs, submits training, parses results, manages datasets."""

    name = "deepmd"
    description = (
        "Manage DeepMD-kit workflow: write input.json, submit training jobs, "
        "parse lcurve.out, test models, prepare NPY datasets from VASP AIMD, "
        "merge datasets, and freeze/compress models."
    )

    def _parameters(self) -> dict:
        """Return the JSON Schema for the LLM tool-call interface."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "write_input_json", "write_sub_sh", "check_training",
                        "parse_lcurve", "test_model", "prepare_dataset",
                        "merge_datasets", "freeze_compress",
                    ],
                },
                "work_dir":          {"type": "string"},
                "type_map":          {"type": "array", "items": {"type": "string"}},
                "n_atoms_per_type":  {"type": "array", "items": {"type": "integer"}},
                "sel":               {"type": "array", "items": {"type": "integer"}},
                "n_steps":           {"type": "integer"},
                "seed":              {"type": "integer"},
                "batch_size":        {"type": "integer"},
                "job_name":          {"type": "string"},
                "walltime":          {"type": "string"},
                "account":           {"type": "string"},
                "test_data_dir":     {"type": "string"},
                "n_frames":          {"type": "integer"},
                "aimd_dirs":         {"type": "array", "items": {"type": "string"}},
                "output_dir":        {"type": "string"},
                "mobile_ion":        {"type": "string"},
                "skip_frac":         {"type": "number"},
                "n_train":           {"type": "number"},
                "dataset_dirs":      {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action"],
        }

    # ── Public methods ────────────────────────────────────────────────────────

    def write_input_json(
        self,
        work_dir: str,
        type_map: list[str],
        n_atoms_per_type: Optional[list[int]] = None,
        sel: Optional[list[int]] = None,
        n_steps: int = 1_000_000,
        seed: int = 42,
        batch_size: int = 32,
        rcut: float = 6.0,
    ) -> Path:
        """
        Write deepmd_input.json with se_e2_a descriptor.
        sel: neighbour list cutoff counts per type. Auto-computed from
             n_atoms_per_type if not provided (2× count + buffer).
        """
        import copy
        inp = copy.deepcopy(_DEEPMD_INPUT_TEMPLATE)
        inp["model"]["type_map"] = type_map

        if sel is None and n_atoms_per_type:
            # Heuristic: 2 × max atoms of that type, rounded up to nearest 10
            sel = [max(10, ((2 * n + 9) // 10) * 10) for n in n_atoms_per_type]
        if sel:
            inp["model"]["descriptor"]["sel"] = sel
        else:
            # Default: generous buffer for up to 3 types
            inp["model"]["descriptor"]["sel"] = [46, 92, 46][: len(type_map)]

        inp["model"]["descriptor"]["rcut"] = rcut
        inp["model"]["descriptor"]["seed"] = seed
        inp["model"]["fitting_net"]["seed"] = seed
        inp["training"]["stop_batch"] = n_steps
        inp["training"]["training_data"]["batch_size"] = batch_size
        inp["training"]["validation_data"]["batch_size"] = batch_size

        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = d / "deepmd_input.json"
        out.write_text(json.dumps(inp, indent=2))
        return out

    def write_sub_sh(
        self,
        work_dir: str,
        job_name: str = "deepmd_train",
        walltime: str = "48:00:00",
        account: str = "",
    ) -> Path:
        """Write GPU submission script: dp train → freeze → compress."""
        account = account or _account_fallback()
        d = Path(work_dir)
        d.mkdir(parents=True, exist_ok=True)
        sub = d / "sub_deepmd.sh"
        content = f"""\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time={walltime}
#SBATCH --job-name={job_name}
#SBATCH --error=%J.stderr
#SBATCH --output=%J.stdout

{_mbl('gpu_md').strip()}
source activate {DP_ENV}
export USE_TF=0
export USE_JAX=0

dp train deepmd_input.json 2>&1 | tee train.log
dp freeze -o pot.pb
dp compress -i pot.pb -o pot_com.pb
"""
        sub.write_text(content)
        sub.chmod(0o755)
        return sub

    def check_training(self, work_dir: str) -> dict:
        """
        Parse the last line of lcurve.out.
        Returns {step, e_rmse, f_rmse, converged, loss}.
        Converged = f_rmse < 100 meV/Å and e_rmse < 5 meV/atom.
        """
        lcurve = Path(work_dir) / "lcurve.out"
        if not lcurve.exists():
            return {"step": 0, "e_rmse": None, "f_rmse": None,
                    "converged": False, "loss": None}

        last = None
        for line in lcurve.read_text(errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                last = line

        if last is None:
            return {"step": 0, "e_rmse": None, "f_rmse": None,
                    "converged": False, "loss": None}

        parts = last.split()
        try:
            # lcurve.out columns (dp2 format):
            # step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  lr
            step    = int(parts[0])
            e_rmse  = float(parts[3]) if len(parts) > 3 else None
            f_rmse  = float(parts[5]) if len(parts) > 5 else None
            loss    = float(parts[1]) if len(parts) > 1 else None
        except (ValueError, IndexError):
            return {"step": 0, "e_rmse": None, "f_rmse": None,
                    "converged": False, "loss": None}

        # Convert from eV to meV for threshold comparison
        e_rmse_mev = (e_rmse * 1000) if e_rmse else None
        f_rmse_mevA = (f_rmse * 1000) if f_rmse else None
        converged = bool(
            e_rmse_mev and f_rmse_mevA
            and e_rmse_mev < 5.0
            and f_rmse_mevA < 100.0
        )

        return {
            "step":      step,
            "e_rmse":    e_rmse,
            "f_rmse":    f_rmse,
            "converged": converged,
            "loss":      loss,
        }

    def parse_lcurve(self, work_dir: str) -> list[dict]:
        """
        Parse full lcurve.out.
        Returns list of {step, rmse_e, rmse_f, rmse_v, lr}.
        """
        lcurve = Path(work_dir) / "lcurve.out"
        if not lcurve.exists():
            return []

        records = []
        for line in lcurve.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                records.append({
                    "step":   int(parts[0]),
                    "rmse_e": float(parts[3]) if len(parts) > 3 else None,
                    "rmse_f": float(parts[5]) if len(parts) > 5 else None,
                    "rmse_v": float(parts[7]) if len(parts) > 7 else None,
                    "lr":     float(parts[-1]) if parts else None,
                })
            except (ValueError, IndexError):
                continue
        return records

    def test_model(
        self,
        work_dir: str,
        test_data_dir: str,
        n_frames: Optional[int] = None,
    ) -> dict:
        """
        Run dp test and parse output.
        Returns {e_rmse, f_rmse, r2}.
        """
        pot = Path(work_dir) / "pot_com.pb"
        if not pot.exists():
            pot = Path(work_dir) / "pot.pb"
        if not pot.exists():
            return {"e_rmse": None, "f_rmse": None, "r2": None,
                    "error": "No model found (pot.pb or pot_com.pb)"}

        cmd = [
            DP_BIN, "test",
            "-m", str(pot),
            "-s", test_data_dir,
            "-n", str(n_frames or 10000),
            "-d", str(Path(work_dir) / "dp_test_results"),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=str(work_dir),
            )
            output = result.stdout + result.stderr
        except Exception as exc:
            return {"e_rmse": None, "f_rmse": None, "r2": None, "error": str(exc)}

        e_rmse = f_rmse = r2 = None
        for line in output.splitlines():
            if "Energy RMSE" in line or "rmse_e" in line.lower():
                m = __import__("re").search(r"[-+]?\d+\.?\d*[eE]?[-+]?\d*", line)
                if m:
                    e_rmse = float(m.group())
            if "Force  RMSE" in line or "rmse_f" in line.lower():
                m = __import__("re").search(r"[-+]?\d+\.?\d*[eE]?[-+]?\d*", line)
                if m:
                    f_rmse = float(m.group())
        return {"e_rmse": e_rmse, "f_rmse": f_rmse, "r2": r2}

    def prepare_dataset(
        self,
        aimd_dirs: list[str],
        output_dir: str,
        mobile_ion: str = "Li",
        skip_frac: float = 0.2,
        n_train: float = 0.9,
    ) -> dict:
        """
        Extract frames from VASP AIMD XDATCAR+OUTCAR and write DeepMD NPY dataset.
        Directories layout: output_dir/set.000/ (train) + validation_data/ (val).
        Returns {n_frames_total, n_train, n_val, dataset_dir}.
        """
        import numpy as np

        out = Path(output_dir)
        set_dir = out / "set.000"
        val_dir = out / "validation_data"
        set_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        all_coords = []
        all_boxes  = []
        all_energies = []
        all_forces = []

        for aimd_dir in aimd_dirs:
            d = Path(aimd_dir)
            xdatcar = d / "XDATCAR"
            outcar  = d / "OUTCAR"
            if not xdatcar.exists():
                continue

            frames, boxes = _parse_xdatcar(xdatcar)
            energies, forces = _parse_outcar_ef(outcar, len(frames))

            # Skip equilibration frames
            skip_n = max(1, int(len(frames) * skip_frac))
            frames    = frames[skip_n:]
            boxes     = boxes[skip_n:]
            energies  = energies[skip_n:] if len(energies) >= len(frames) + skip_n else energies
            forces     = forces[skip_n:]   if len(forces)   >= len(frames) + skip_n else forces

            min_len = min(len(frames), len(energies), len(forces))
            frames   = frames[:min_len]
            boxes    = boxes[:min_len]
            energies = energies[:min_len]
            forces   = forces[:min_len]

            all_coords.extend(frames)
            all_boxes.extend(boxes)
            all_energies.extend(energies)
            all_forces.extend(forces)

        n_total = len(all_coords)
        if n_total == 0:
            return {"n_frames_total": 0, "n_train": 0, "n_val": 0,
                    "dataset_dir": str(out), "error": "No frames extracted"}

        split = int(n_total * n_train)
        train_idx = slice(0, split)
        val_idx   = slice(split, None)

        def _save(directory, coords, boxes, energies, forces):
            """Save coords, boxes, energies, and forces as DeepMD NPY arrays in directory."""
            np.save(directory / "coord.npy",  np.array(coords,    dtype=np.float64))
            np.save(directory / "box.npy",    np.array(boxes,     dtype=np.float64))
            np.save(directory / "energy.npy", np.array(energies,  dtype=np.float64))
            np.save(directory / "force.npy",  np.array(forces,    dtype=np.float64))

        _save(set_dir,
              all_coords[train_idx], all_boxes[train_idx],
              all_energies[train_idx], all_forces[train_idx])
        _save(val_dir,
              all_coords[val_idx], all_boxes[val_idx],
              all_energies[val_idx], all_forces[val_idx])

        return {
            "n_frames_total": n_total,
            "n_train":        split,
            "n_val":          n_total - split,
            "dataset_dir":    str(out),
        }

    def merge_datasets(self, dataset_dirs: list[str], output_dir: str) -> Path:
        """Concatenate multiple DeepMD NPY datasets into one."""
        import numpy as np

        out = Path(output_dir) / "set.000"
        out.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, list] = {
            "coord": [], "box": [], "energy": [], "force": []
        }
        for ds in dataset_dirs:
            d = Path(ds)
            for key in arrays:
                npy = d / f"{key}.npy"
                if npy.exists():
                    arrays[key].append(np.load(str(npy)))

        for key, parts in arrays.items():
            if parts:
                merged = np.concatenate(parts, axis=0)
                np.save(str(out / f"{key}.npy"), merged)

        return out

    def freeze_compress(self, work_dir: str) -> ToolResult:
        """Run dp freeze → dp compress inside work_dir."""
        d = Path(work_dir)
        cmds = [
            f"{DP_BIN} freeze -o pot.pb",
            f"{DP_BIN} compress -i pot.pb -o pot_com.pb",
        ]
        outputs = []
        for cmd in cmds:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=300, cwd=str(d),
                )
                outputs.append(f"$ {cmd}\n" + (result.stdout or result.stderr))
                if result.returncode != 0:
                    return ToolResult(
                        "\n".join(outputs),
                        success=False,
                        metadata={"returncode": result.returncode},
                    )
            except Exception as exc:
                return ToolResult(str(exc), success=False)

        return ToolResult("\n".join(outputs), metadata={"pot": str(d / "pot_com.pb")})

    # ── execute() dispatch ─────────────────────────────────────────────────────

    def execute(self, action: str, **kwargs) -> ToolResult:
        """Dispatch an LLM tool-call action and return a ToolResult."""
        try:
            if action == "write_input_json":
                p = self.write_input_json(
                    kwargs["work_dir"],
                    type_map=kwargs.get("type_map", []),
                    n_atoms_per_type=kwargs.get("n_atoms_per_type"),
                    sel=kwargs.get("sel"),
                    n_steps=kwargs.get("n_steps", 1_000_000),
                    seed=kwargs.get("seed", 42),
                    batch_size=kwargs.get("batch_size", 32),
                )
                return ToolResult(f"deepmd_input.json written: {p}")

            elif action == "write_sub_sh":
                p = self.write_sub_sh(
                    kwargs["work_dir"],
                    job_name=kwargs.get("job_name", "deepmd_train"),
                    walltime=kwargs.get("walltime", "48:00:00"),
                    account=kwargs.get("account") or _account_fallback(),
                )
                return ToolResult(f"sub_deepmd.sh written: {p}")

            elif action == "check_training":
                d = self.check_training(kwargs["work_dir"])
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "parse_lcurve":
                records = self.parse_lcurve(kwargs["work_dir"])
                if not records:
                    return ToolResult("No lcurve.out found or empty.")
                last = records[-1]
                return ToolResult(
                    f"Steps: {last['step']}  E_rmse={last['rmse_e']}  "
                    f"F_rmse={last['rmse_f']}  lr={last['lr']}",
                    metadata={"n_records": len(records), "last": last},
                )

            elif action == "test_model":
                d = self.test_model(
                    kwargs["work_dir"],
                    kwargs.get("test_data_dir", "00.data/validation_data"),
                    n_frames=kwargs.get("n_frames"),
                )
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "prepare_dataset":
                d = self.prepare_dataset(
                    kwargs.get("aimd_dirs", []),
                    kwargs["output_dir"],
                    mobile_ion=kwargs.get("mobile_ion", "Li"),
                    skip_frac=kwargs.get("skip_frac", 0.2),
                    n_train=kwargs.get("n_train", 0.9),
                )
                text = "\n".join(f"{k}: {v}" for k, v in d.items())
                return ToolResult(text, metadata=d)

            elif action == "merge_datasets":
                p = self.merge_datasets(
                    kwargs.get("dataset_dirs", []),
                    kwargs["output_dir"],
                )
                return ToolResult(f"Merged dataset at: {p}")

            elif action == "freeze_compress":
                return self.freeze_compress(kwargs["work_dir"])

            else:
                return ToolResult(f"Unknown action: {action}", success=False)

        except Exception as exc:
            return ToolResult(str(exc), success=False)


# ── Helper parsers (module-level, not part of the class) ─────────────────────

def _parse_xdatcar(xdatcar_path: Path):
    """
    Parse VASP XDATCAR into list of coordinate arrays and box arrays.
    Returns (frames, boxes) where each element is a numpy ndarray.
    """
    import numpy as np

    text = xdatcar_path.read_text(errors="replace").splitlines()
    idx = 0
    # Line 0: system name; Line 1: scale; Lines 2-4: lattice; 5: species; 6: counts
    scale = float(text[1].split()[0]) if len(text) > 1 else 1.0
    a1 = [float(x) * scale for x in text[2].split()]
    a2 = [float(x) * scale for x in text[3].split()]
    a3 = [float(x) * scale for x in text[4].split()]
    box = np.array([a1, a2, a3])  # (3,3)
    counts = list(map(int, text[6].split()))
    n_atoms = sum(counts)

    frames = []
    boxes  = []
    i = 7
    while i < len(text):
        line = text[i].strip()
        if line.startswith("Direct") or line.startswith("Cartesian"):
            i += 1
            frame_frac = []
            for _ in range(n_atoms):
                if i < len(text):
                    parts = text[i].split()
                    frame_frac.append([float(parts[j]) for j in range(3)])
                    i += 1
            # Convert fractional → Cartesian
            frac = np.array(frame_frac)         # (n_atoms, 3)
            cart = frac @ box                   # (n_atoms, 3)
            frames.append(cart.flatten())       # shape (3*n_atoms,)
            # Box as 9-element flat array (row-major)
            boxes.append(box.flatten())
        else:
            i += 1

    return frames, boxes


def _parse_outcar_ef(outcar_path: Path, n_expected: int):
    """
    Parse energies and forces from VASP OUTCAR.
    Returns (energies, forces) lists.
    energies: list of float (eV)
    forces:   list of flat arrays (3*n_atoms,)
    """
    if not outcar_path.exists():
        return [], []

    text = outcar_path.read_text(errors="replace")
    import re
    import numpy as np

    # Energy: "energy  without entropy=" lines
    energies = [
        float(m.group(1))
        for m in re.finditer(
            r"energy\s+without entropy=\s+([-\d.E+]+)", text
        )
    ]

    # Forces: blocks after "TOTAL-FORCE (eV/Angst)"
    forces = []
    for m in re.finditer(r"TOTAL-FORCE \(eV/Angst\)\s*\n-+\n(.*?)\n-+", text, re.DOTALL):
        block = m.group(1).strip().splitlines()
        frame_forces = []
        for line in block:
            parts = line.split()
            if len(parts) >= 6:
                frame_forces += [float(parts[3]), float(parts[4]), float(parts[5])]
        if frame_forces:
            forces.append(np.array(frame_forces))

    return energies, forces
