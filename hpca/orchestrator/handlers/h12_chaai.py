"""
h12_chaai.py — CHAAI LLM training pipeline handler.

Mode A (daemon): generates Qwen-format training examples on every stage completion.
Mode B (SLURM): submits 5-stage training pipeline when accumulated examples > threshold.

Chaai root: /path/to/apps/Chaai/
Training pipeline: collect → synthesize → sft_train → verify → dpo_train
Done: adapters/chaai-v1/adapter_model.bin exists
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

from hpca.core.paths import load_platform_config as _lpc
CHAAI_ROOT = Path(_lpc().get("hpc", {}).get("chaai_root", "Chaai"))
CHAAI_DATA = CHAAI_ROOT / "training" / "data"
CHAAI_ADAPTERS = CHAAI_ROOT / "adapters"
NEW_EXAMPLES_THRESHOLD = 500

# System prompt used in all examples
SYSTEM_PROMPT = (
    "You are CHAAI, an expert AI assistant for computational materials science on NREL Kestrel HPC. "
    "You specialize in VASP DFT, DeepMD machine learning potentials, LAMMPS molecular dynamics, "
    "NEB calculations, and battery materials characterization. Provide precise, executable answers."
)


class CHAAIHandler(SimulationHandler):
    """Handler: generates training examples and submits CHAAI fine-tuning pipeline."""

    name = "h12_chaai"
    is_daemon = True  # runs in-process; sbatch only when training threshold hit

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Always returns True; CHAAI example generation runs alongside every other handler."""
        return True  # Always runs

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when the chaai-v1 adapter exists, or False while the training pipeline is running."""
        # Complete when adapter exists OR when we've already submitted the pipeline
        if (CHAAI_ADAPTERS / "chaai-v1" / "adapter_model.bin").exists():
            return True
        handler_state = state.get_handler("h12_chaai")
        # If pipeline submitted, stay RUNNING until adapter appears
        if handler_state.get("pipeline_submitted"):
            return False
        # Otherwise "complete" immediately — ongoing generation via _chaai_on_complete()
        return True

    def on_stage_complete(self, handler_name: str, project_dir: Path,
                          state: "ProjectState") -> None:
        """Called when any handler completes — generates a CHAAI training example."""
        try:
            self._generate_example(project_dir, state)
        except Exception as exc:
            log.debug("[h12_chaai] on_stage_complete failed (%s): %s", handler_name, exc)

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Daemon submit: generate one example; kick off training when threshold met."""
        self._generate_example(project_dir, state)
        self._maybe_submit_training(state)
        return None  # daemon — no job_id

    def _maybe_submit_training(self, state: "ProjectState") -> None:
        """Submit training pipeline if example count exceeds threshold."""
        count = state.get_chaai_count()
        log.info("[h12_chaai] CHAAI example count: %d / %d", count, NEW_EXAMPLES_THRESHOLD)
        if count < NEW_EXAMPLES_THRESHOLD:
            return
        handler_state = state.get_handler("h12_chaai")
        pipeline_jobs = handler_state.get("pipeline_jobs", {})
        if pipeline_jobs:
            any_alive = any(self.job_alive(jid) for jid in pipeline_jobs.values() if jid)
            if any_alive:
                log.info("[h12_chaai] Training pipeline already running")
                return
        self._submit_training_pipeline(state)

    def _generate_example(self, project_dir: Path, state: "ProjectState") -> None:
        """Generate one training example based on recently completed stages."""
        yaml = self.read_project_yaml(project_dir)
        project_name = yaml.get("name", project_dir.name)

        # Detect which stage just completed and generate appropriate example
        examples_generated = 0

        # Check handler stages to find what has recently completed
        h_state = state.get_handler("h12_chaai")
        last_stages = h_state.get("last_seen_stages", {})

        stage_to_maker = {
            "h02_aimd":  self._make_aimd_example,
            "h06_analysis": self._make_msd_example,
            "h04_mlip":  self._make_mlip_example,
            "h03_neb":   self._make_neb_example,
            "h01_dft":   self._make_dft_example,
            "h05_lammps": self._make_lammps_example,
        }

        for handler_name, maker in stage_to_maker.items():
            current_stage = state.get_stage(handler_name)
            prev_stage = last_stages.get(handler_name, "PENDING")

            if current_stage == "COMPLETE" and prev_stage != "COMPLETE":
                try:
                    example = maker(project_dir, state, project_name, yaml)
                    if example:
                        self._write_example(example)
                        state.increment_chaai_examples(1)
                        examples_generated += 1
                        log.debug("[h12_chaai] Generated %s example for %s",
                                  handler_name, project_name)
                except Exception as exc:
                    log.debug("[h12_chaai] Example generation failed for %s: %s",
                              handler_name, exc)

        # Update last seen stages
        new_last = dict(last_stages)
        for handler_name in stage_to_maker:
            new_last[handler_name] = state.get_stage(handler_name)
        state.set_handler("h12_chaai", {"last_seen_stages": new_last})

        if examples_generated == 0:
            # Generate a generic project-overview example if nothing just completed
            try:
                example = self._make_overview_example(project_dir, state, project_name, yaml)
                if example:
                    self._write_example(example)
                    state.increment_chaai_examples(1)
            except Exception as exc:
                log.debug("[h12_chaai] Overview example failed: %s", exc)

    # ── Example makers ──────────────────────────────────────────────────────

    def _make_aimd_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_a_simulation: VASP AIMD submission workflow."""
        temps = yaml.get("aimd_temps", [300, 600])
        encut = yaml.get("encut_aimd", 400.8232)
        nsw = yaml.get("nsw_aimd", 50000)
        mobile_ion = yaml.get("mobile_ion", "Li")
        formula = yaml.get("formula", project_name)

        proj_path = str(project_dir)
        user_msg = (
            f"I have a relaxed {formula} structure in {proj_path}/dft/opt/CONTCAR. "
            f"Please set up and submit VASP NVT AIMD jobs at {temps} K "
            f"for {nsw} steps with a 1 fs timestep."
        )
        assistant_msg = (
            f"Here is the workflow to set up {formula} AIMD at {temps} K:\n\n"
            "```bash\n"
            f"# For each temperature:\n"
        )
        for T in temps[:2]:
            assistant_msg += (
                f"mkdir -p {proj_path}/aimd/{T}\n"
                f"cd {proj_path}/aimd/{T}\n"
                f"cp ../../opt/CONTCAR POSCAR\n"
                f"cp ../../opt/POTCAR .\n"
            )
        assistant_msg += "```\n\n"
        assistant_msg += "INCAR settings (NVT AIMD):\n```\n"
        assistant_msg += (
            f"SYSTEM = {project_name}_AIMD_{temps[0]}K\n"
            f"PREC   = Medium\n"
            f"ENCUT  = {encut}\n"
            "ISMEAR = 0\nSIGMA  = 0.02\n"
            "NELM   = 60\nEDIFF  = 1E-4\nNELMIN = 4\n"
            "IBRION = 0\nPOTIM  = 1.0\n"
            f"NSW    = {nsw}\n"
            f"ISIF   = 2\nSMASS  = 0\nISYM   = 0\n"
            f"TEBEG  = {temps[0]}\nTEEND  = {temps[0]}\n"
            "LWAVE  = F\nLCHARG = F\nLREAL  = A\nNPAR   = 8\n"
        )
        assistant_msg += "```\n"
        assistant_msg += (
            "Submit with:\n```bash\nsbatch sub.sh\n```\n"
            "Monitor steps: `grep 'T=' OSZICAR | wc -l`\n"
            "Check for NaN temperature (NAN_TEMP error): requires LREAL=A and no NELMDL=-12."
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_a_simulation",
            source=f"hpca/{project_name}/h02_aimd",
        )

    def _make_msd_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_b_analysis: MSD and diffusivity from LAMMPS dump."""
        mobile_ion = yaml.get("mobile_ion", "Li")
        temps = yaml.get("mlmd_temps", yaml.get("aimd_temps", [300, 600]))

        # Try to read actual Ea from results
        Ea_str = "~0.3 eV (typical for fast-ion conductors)"
        _aimd_csv = project_dir / "Analysis" / "aimd" / "arrhenius.csv"
        _candidates = sorted(project_dir.glob("Analysis/*/arrhenius.csv"))
        arrh_csv = _aimd_csv if _aimd_csv.exists() else (_candidates[0] if _candidates else None)
        if arrh_csv is not None and arrh_csv.exists():
            try:
                import numpy as np
                data = np.loadtxt(str(arrh_csv), delimiter=",", skiprows=1)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                if data.shape[1] > 4:
                    Ea_str = f"{data[0, 4]:.3f} eV (computed)"
            except Exception:
                pass

        user_msg = (
            f"How do I compute {mobile_ion}-ion diffusivity and activation energy from "
            f"LAMMPS dump files at {temps} K for {project_name}?"
        )
        assistant_msg = (
            f"Use hpca's MSD analysis (hpca.analysis.msd) or the hpca analysis scripts:\n\n"
            "```python\nimport numpy as np\nfrom scipy import stats\n"
            "from pathlib import Path\n\n"
            "def parse_dump_li(fpath, element='Li'):\n"
            "    # Reads dump_unwrapped.lmp, extracts target-element positions\n"
            "    # Returns (n_frames, n_Li, 3) array\n"
            "    ...\n\n"
            "def compute_msd(positions, dt_ps=1.0):\n"
            "    skip = int(len(positions) * 0.2)\n"
            "    pos = positions[skip:]\n"
            "    n = len(pos)\n"
            "    max_lag = int(n * 0.5)\n"
            "    msd = np.array([\n"
            "        np.mean((pos[lag:] - pos[:n-lag])**2) * 3\n"
            "        for lag in range(1, max_lag+1)\n"
            "    ])\n"
            "    times = np.arange(1, max_lag+1) * dt_ps\n"
            "    lo, hi = int(0.4*max_lag), int(0.8*max_lag)\n"
            "    slope, *_ = stats.linregress(times[lo:hi], msd[lo:hi])\n"
            "    D = slope / 6.0 * 1e-8  # Å²/ps → m²/s\n"
            "    return times, msd, D\n"
            "```\n\n"
            f"Arrhenius: `Ea = -slope * 8.617333e-5` eV where slope = d(lnD)/d(1/T).\n"
            f"For {project_name}: Ea ≈ {Ea_str}.\n"
            f"Output paths: {project_dir}/results/data/arrhenius.csv and .png"
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_b_analysis",
            source=f"hpca/{project_name}/h06_analysis",
        )

    def _make_mlip_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_e_mlip: DeepMD training workflow."""
        type_map = yaml.get("type_map", ["Li", "Cl"])
        user_msg = (
            f"Train a DeepMD potential for {project_name} (elements: {type_map}) "
            f"on the AIMD dataset at {project_dir}/mlmd/mlff/00.data/"
        )
        assistant_msg = (
            f"DeepMD training for {project_name}:\n\n"
            "```bash\n"
            f"cd {project_dir}/mlmd/mlff\n"
            "# Generate deepmd_input.json with type_map and sel for your elements\n"
            "sbatch sub_deepmd.sh\n"
            "```\n\n"
            "Key deepmd_input.json fields:\n"
            "```json\n"
            '{\n  "model": {\n'
            f'    "type_map": {json.dumps(type_map)},\n'
            '    "descriptor": {"type": "se_a", "rcut": 6.0, "neuron": [25, 50, 100]},\n'
            '    "fitting_net": {"neuron": [240, 240, 240]}\n'
            '  },\n'
            '  "training": {"numb_steps": 500000}\n'
            '}\n```\n\n'
            "After training:\n"
            "```bash\ndp freeze -o pot.pb\ndp compress -i pot.pb -o pot_com.pb\n"
            "dp test -m pot_com.pb -s 00.data -n 1000 -d test_results\n```\n"
            "Acceptance: E RMSE < 5 meV/atom, F RMSE < 100 meV/Å.\n"
            "GPU: 4× H100 on gpu-h100 partition, ~6–12 h for 500k steps."
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_e_mlip",
            source=f"hpca/{project_name}/h04_mlip",
        )

    def _make_neb_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_c_hpc: NEB barrier analysis."""
        mobile_ion = yaml.get("mobile_ion", "Li")

        # Try to read actual barriers
        barriers_str = "not yet computed"
        neb_json = project_dir / "results" / "neb_barriers.json"
        if neb_json.exists():
            try:
                neb_data = json.loads(neb_json.read_text())
                vals = [f"{v['Ea_meV']:.0f} meV" for v in neb_data.values()
                        if isinstance(v, dict) and v.get("Ea_meV")]
                if vals:
                    barriers_str = f"Ea = {', '.join(vals)}"
            except Exception:
                pass

        user_msg = (
            f"What is the {mobile_ion}-ion migration barrier in {project_name} "
            f"and how do I set up NEB calculations?"
        )
        assistant_msg = (
            f"NEB migration barrier for {project_name}: {barriers_str}.\n\n"
            "Setup with ASE IDPP interpolation (NOT nebmake.pl to avoid atom clashes):\n"
            "```python\nfrom ase.io import read, write\n"
            "from ase.neb import NEB\n"
            "from ase.optimize import FIRE\n\n"
            "initial = read('POSCAR_ini')\nfinal = read('POSCAR_fin')\n"
            "images = [initial.copy() for _ in range(11)] + [final.copy()]\n"
            "neb = NEB(images)\nneb.interpolate('idpp')  # IDPP avoids atom clashes\n"
            "for i, img in enumerate(images[1:-1], 1):\n"
            "    write(f'{i:02d}/POSCAR', img)\n```\n\n"
            "INCAR (NEB): IMAGES=11, LCLIMB=.TRUE., SPRING=-5, EDIFFG=-0.05\n"
            f"Submit: sbatch {project_dir}/neb/neb_submit.sh\n"
            "Extract: nebresults.pl or parse neb_run/*/pipeline_state.json"
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_c_hpc",
            source=f"hpca/{project_name}/h03_neb",
        )

    def _make_dft_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_a_simulation: DFT relaxation and subtask workflow."""
        formula = yaml.get("formula", project_name)
        encut = yaml.get("encut", 520.0)
        user_msg = (
            f"How do I run a full DFT characterization workflow for {formula} on Kestrel?"
        )
        assistant_msg = (
            f"Full DFT workflow for {formula}:\n\n"
            "1. vc-relax (ISIF=3): optimize cell + ions\n"
            f"   `sbatch {project_dir}/dft/vc/sub.sh`\n"
            "   After: `cp CONTCAR ../opt/POSCAR`\n\n"
            "2. opt (ISIF=2): ionic relaxation at fixed cell\n"
            "3. bader: LCHARG=T, LAECHG=T; post-process with `chgsum.pl + bader`\n"
            "4. dos/scf + dos/nonscf: ICHARG=11, NEDOS=2000\n\n"
            f"Key INCAR: ENCUT={encut}, PREC=Accurate, SYMPREC=1E-4\n"
            "KPOINTS: Gamma 2×2×2 for SSEs\n"
            "Submit with: `sbatch sub.sh` (1 node, 96 tasks, 48h)\n"
            "Monitor: `squeue -u $USER`"
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_a_simulation",
            source=f"hpca/{project_name}/h01_dft",
        )

    def _make_lammps_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """domain_a_simulation: LAMMPS MLMD setup."""
        temps = yaml.get("mlmd_temps", [300, 600])
        mobile_ion = yaml.get("mobile_ion", "Li")
        user_msg = (
            f"Set up LAMMPS MLMD for {project_name} at {temps} K using DeepMD potential."
        )
        assistant_msg = (
            "LAMMPS in.lammps skeleton for DeepMD + GPU:\n"
            "```lammps\nunits        metal\natom_style   atomic\nboundary     p p p\n"
            "read_data    data.lammps\n\n"
            "pair_style   deepmd pot_com.pb\npair_coeff   * *\n\n"
            "timestep     0.001\nthermo       1000\n"
            "thermo_style custom step temp pe ke etotal press\n\n"
            "dump  1 all custom 1000 dump_unwrapped.lmp id type xu yu zu\n"
            f"dump_modify  1 element {mobile_ion} sort id\n\n"
            f"fix   1 all nvt temp 300 300 0.05\nrun   1000000\n```\n\n"
            "GPU submission (4× H100, gpu-h100 partition):\n"
            "```bash\n#SBATCH --gpus=4 --ntasks-per-node=4\n"
            "srun --gpus-per-task=1 lmp -k on gpus 4 -sf kk -in in.lammps\n```\n"
            "After run: check dump_unwrapped.lmp size > 1 MB (valid trajectory)."
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_a_simulation",
            source=f"hpca/{project_name}/h05_lammps",
        )

    def _make_overview_example(
        self, project_dir: Path, state: "ProjectState",
        project_name: str, yaml: dict
    ) -> dict | None:
        """Generic project overview example."""
        category = yaml.get("category", "inorganic_sse")
        stages_done = [h for h in [
            "h00_design", "h01_dft", "h02_aimd", "h03_neb",
            "h04_mlip", "h05_lammps", "h06_analysis",
        ] if state.get_stage(h) == "COMPLETE"]

        user_msg = (
            f"Summarize the computational workflow status for {project_name} "
            f"(category: {category})."
        )
        assistant_msg = (
            f"Workflow status for {project_name}:\n"
            f"- Category: {category}\n"
            f"- Completed stages: {', '.join(stages_done) if stages_done else 'none yet'}\n"
            f"- Project directory: {project_dir}/\n\n"
            "Standard pipeline:\n"
            "00_design → 01_dft → 02_aimd → 04_mlip → 05_lammps → 06_analysis → "
            "07_electronic → 08_echem → 09_continuum → 10_plotting → 11_manuscript\n\n"
            "Check status: look for CONTCAR (DFT done), energy.npy (AIMD done), "
            "pot_com.pb (MLIP done), dump_unwrapped.lmp (MLMD done), "
            "arrhenius.csv (analysis done)."
        )

        return _make_chat_example(
            user_msg, assistant_msg,
            task_type="domain_d_general",
            source=f"hpca/{project_name}/overview",
        )

    # ── Data writing ────────────────────────────────────────────────────────

    def _write_example(self, example: dict) -> None:
        """Append example to appropriate JSONL file in CHAAI_DATA after validation."""
        task_type = example.get("task_type", "domain_a_simulation")
        # Validate: must have messages list with at least one user turn
        messages = example.get("messages", [])
        if not isinstance(messages, list) or not messages:
            log.warning("[h12_chaai] Skipping example with invalid messages: %s", task_type)
            return
        if not any(m.get("role") == "user" for m in messages):
            log.warning("[h12_chaai] Skipping example with no user turn: %s", task_type)
            return
        try:
            line = json.dumps(example)
        except (TypeError, ValueError) as exc:
            log.warning("[h12_chaai] Skipping non-serialisable example: %s", exc)
            return
        CHAAI_DATA.mkdir(parents=True, exist_ok=True)
        out_file = CHAAI_DATA / f"{task_type}.jsonl"
        with open(out_file, "a") as fh:
            fh.write(line + "\n")

    # ── Training pipeline submission ────────────────────────────────────────

    def _submit_training_pipeline(self, state: "ProjectState") -> str | None:
        """Submit 5-stage CHAAI training pipeline with SLURM dependencies."""
        training_dir = CHAAI_ROOT / "training"

        scripts = {
            "collect":    training_dir / "collect_submit.sh",
            "synthesize": training_dir / "synthesize_submit.sh",
            "sft_train":  training_dir / "train_submit.sh",
            "dpo_train":  training_dir / "dpo_submit.sh",
        }

        # Check that scripts exist
        missing = [str(s) for s in scripts.values() if not s.exists()]
        if missing:
            log.warning("[h12_chaai] Training scripts not found: %s — skipping pipeline", missing)
            return None

        pipeline_jobs: dict[str, str | None] = {}

        # Stage 1: collect
        j1 = self.sbatch(scripts["collect"], cwd=training_dir)
        pipeline_jobs["collect"] = j1
        if not j1:
            log.error("[h12_chaai] Failed to submit collect job")
            return None
        log.info("[h12_chaai] Submitted collect job=%s", j1)

        # Stage 2: synthesize (after collect)
        j2 = self.sbatch(scripts["synthesize"],
                         cwd=training_dir,
                         extra_args=[f"--dependency=afterok:{j1}"])
        pipeline_jobs["synthesize"] = j2
        log.info("[h12_chaai] Submitted synthesize job=%s (dep=%s)", j2, j1)

        # Stage 3: sft_train (after synthesize)
        j3 = self.sbatch(scripts["sft_train"],
                         cwd=training_dir,
                         extra_args=[f"--dependency=afterok:{j2}"]) if j2 else None
        pipeline_jobs["sft_train"] = j3
        log.info("[h12_chaai] Submitted sft_train job=%s (dep=%s)", j3, j2)

        # Stage 4: dpo_train (after sft_train)
        j4 = self.sbatch(scripts["dpo_train"],
                         cwd=training_dir,
                         extra_args=[f"--dependency=afterok:{j3}"]) if j3 else None
        pipeline_jobs["dpo_train"] = j4
        log.info("[h12_chaai] Submitted dpo_train job=%s (dep=%s)", j4, j3)

        state.set_handler("h12_chaai", {
            "pipeline_jobs": pipeline_jobs,
            "pipeline_submitted": datetime.now().isoformat(),
        })
        log.info("[h12_chaai] Training pipeline submitted: %s", pipeline_jobs)

        return j1  # Return first job id

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Log example count or training pipeline job status; trigger training when threshold is crossed."""
        handler_state = state.get_handler("h12_chaai")
        pipeline_jobs = handler_state.get("pipeline_jobs", {})

        if not pipeline_jobs:
            count = state.get_chaai_count()
            log.info("[h12_chaai] Examples collected: %d / %d", count, NEW_EXAMPLES_THRESHOLD)
            # Trigger training pipeline if threshold just crossed
            self._maybe_submit_training(state)
            return

        for stage, job_id in pipeline_jobs.items():
            if job_id:
                alive = self.job_alive(job_id)
                log.info("[h12_chaai] Pipeline stage %s job=%s alive=%s",
                         stage, job_id, alive)
            else:
                log.warning("[h12_chaai] Pipeline stage %s: no job ID", stage)

    def on_complete(self, project_dir: Path, state: "ProjectState") -> None:
        """Record completion timestamp and adapter path in state when the adapter is ready."""
        adapter_path = CHAAI_ADAPTERS / "chaai-v1"
        if adapter_path.exists():
            log.info("[h12_chaai] CHAAI adapter ready at %s", adapter_path)
        else:
            log.info("[h12_chaai] Data collection active; training pipeline triggers at %d examples",
                     NEW_EXAMPLES_THRESHOLD)
        state.set_handler("h12_chaai", {
            "completed": datetime.now().isoformat(),
            "adapter_path": str(adapter_path),
        })


# ── Module-level helper ───────────────────────────────────────────────────────────

def _make_chat_example(
    user_msg: str, assistant_msg: str,
    task_type: str = "domain_a_simulation",
    source: str = "hpca",
) -> dict:
    """Build a Qwen-format chat training example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "source": source,
        "task_type": task_type,
        "created": datetime.now().isoformat(),
    }
