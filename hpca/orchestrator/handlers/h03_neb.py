"""
h03_neb.py — Constrained NEB handler for inorganic_sse projects.

Methodology: independent geometry optimisation per image with selective dynamics
(migrating atom held at interpolated position, neighbouring atoms free within 4 Å),
NOT VASP CI-NEB mode.  Each image is a normal IBRION=2 run; images execute in
parallel via srun --ntasks=16 --exclusive inside a single SLURM node allocation
(96 cores → 6 images/node).

Starting structure: always dft/opt/CONTCAR (DFT-optimised by h01_dft).  No
separate NEB preopt is needed — h01 already provides a fully relaxed cell.

3-phase orchestration (polling-based, no blocking):
  Phase 1: setup paths + endpoint POSCAR directories, submit sub_endpoints.sh
  Phase 2: poll until endpoints converge, generate images, submit image jobs
  Phase 3: poll until all images converge, extract barriers, write pipeline_state.json

Image generation uses hpca.core.neb (pymatgen-based, nonlinear L-BFGS-B path
predictor with repulsion from framework atoms). Falls back to hpca.core.neb.linear
(pure-numpy linear interpolation) if pymatgen is unavailable.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .base import SimulationHandler
from hpca.core.paths import dft_opt, load_platform_config
from hpca.core.neb import make_neb_images, apply_selective_dynamics, find_migrating_atom
from hpca.core.config import Config as _Config
from hpca.core.potcar import build_potcar as _build_potcar
from hpca.core.vasp_job import (
    read_poscar_elements as _read_poscar_elements,
)
from hpca.registry.submission import write_submission as _write_sub
from hpca.core.kpoints import kpoints_from_poscar as _kp_from_poscar
from hpca.registry.incar import build_incar as _build_incar, write_incar as _write_incar

if TYPE_CHECKING:
    from hpca.orchestrator.state_tracker import ProjectState

log = logging.getLogger("hpca.orch")

_PLATFORM = load_platform_config()
_VASP_MODULE  = _PLATFORM.get("hpc", {}).get("vasp_module",  "vasp/6.4.2_openMP")
_VASP_ACCOUNT = (_PLATFORM.get("hpc", {}).get("vasp_account")
                 or _PLATFORM.get("hpc", {}).get("accounts", {}).get("standard") or _account_fallback())
_CLADUE_SITE  = _PLATFORM.get("hpc", {}).get("cladue_site_packages", "")

_N_IMAGES_DEFAULT = int(_PLATFORM.get("hpc", {}).get("neb_n_images", 11))  # per-project override: yaml neb_images
_CORES_PER_IMAGE  = int(_PLATFORM.get("hpc", {}).get("neb_cores_per_image", 16))
_CORES_PER_NODE   = int(_PLATFORM.get("hpc", {}).get("neb_cores_per_node", 96))
_IMAGES_NODE1     = 6    # images 01-06 on node 1
# images 07-11 on node 2 (5 images, 80 of 96 cores used)

# Halide oxidation potentials vs Li/Li+ (V) used in h08_echem as well
_HAL_VOX = {"F": 6.0, "Cl": 4.3, "Br": 3.5, "I": 2.9}


import sys
from hpca.core.config import account_fallback as _account_fallback


def _ensure_cladue_env() -> None:
    """Prepend the cladue site-packages directory to sys.path if not already present."""
    if _CLADUE_SITE and _CLADUE_SITE not in sys.path:
        sys.path.insert(0, _CLADUE_SITE)


class NEBHandler(SimulationHandler):
    """Constrained NEB: vacancy + interstitial, multiple crystallographic directions."""

    name = "h03_neb"
    is_daemon = False

    # ── Gate checks ──────────────────────────────────────────────────────────

    def can_run(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when NEB is enabled and the DFT-optimised CONTCAR from h01 is present."""
        yaml = self.read_project_yaml(project_dir)
        category = yaml.get("category", "")
        from hpca.core.categories import is_sse as _is_sse
        _is_int = category.startswith("int") or "interface" in category.lower()
        neb_enabled = (
            yaml.get("neb", False)
            or yaml.get("stages", {}).get("neb", False)
            or _is_sse(category)
            or _is_int
        )
        if not neb_enabled:
            return False
        return (dft_opt(project_dir) / "CONTCAR").exists()

    def is_complete(self, project_dir: Path, state: "ProjectState") -> bool:
        """Return True when all NEB paths have COMPLETE stage and a non-None Ea_meV in pipeline_state.json."""
        neb_dir = project_dir / "neb"
        state_file = neb_dir / "pipeline_state.json"
        if not state_file.exists():
            return False
        try:
            ps = json.loads(state_file.read_text())
        except Exception:
            return False
        paths = ps.get("paths", {})
        if not paths:
            return False
        return all(
            info.get("stage") == "COMPLETE" and info.get("Ea_meV") is not None
            for info in paths.values()
            if isinstance(info, dict)
        )

    def auto_fix(self, project_dir: Path, state: "ProjectState") -> bool:
        """
        Prevent spurious FAILED when check_progress() transitions between phases and
        the orchestrator sees the OLD job as dead before recognising the NEW job.
        Also handles dead jobs that need a genuine reset.
        """
        hs = state.get_handler("h03_neb") or {}
        neb_dir = project_dir / "neb"
        phase = hs.get("phase", 1)

        # Phase 2: image jobs (possibly just submitted by _check_phase1 inside check_progress)
        if phase == 2:
            image_jobs = hs.get("image_jobs", [])
            live = [j for j in image_jobs if self.job_alive(j)]
            if live:
                log.info("[h03_neb] auto_fix: phase 2 — %d image jobs alive", len(live))
                state.set_stage("h03_neb", "RUNNING", job=live[0])
                return True
            # All image jobs dead: if all images converged, phase 3 will run next poll
            paths = hs.get("paths", [])
            n_images = hs.get("n_images", _N_IMAGES_DEFAULT)
            all_done = paths and all(
                self._outcar_converged(neb_dir / pname / f"{img:02d}" / "OUTCAR")
                for pname in paths
                for img in range(1, n_images + 1)
            )
            if all_done:
                log.info("[h03_neb] auto_fix: all images converged — keeping RUNNING for phase 3")
                state.set_stage("h03_neb", "RUNNING")
                return True
            if not image_jobs:
                # _phase2_images() was interrupted (e.g., by SIGTERM mid-run) — rewind to
                # phase 1 so check_progress() re-runs it in full on the next poll.
                log.warning("[h03_neb] auto_fix: phase 2 with no image_jobs — rewinding to phase 1")
                state.set_handler("h03_neb", {**hs, "phase": 1})
                state.set_stage("h03_neb", "RUNNING")
                return True
            log.warning("[h03_neb] auto_fix: image jobs dead, not converged — reset to PENDING")
            state.set_stage("h03_neb", "PENDING")
            state.set_handler("h03_neb", {})
            return True

        # Phase 1: endpoint jobs (or just transitioned from preopt "done")
        if phase == 1:
            ep_jobs = hs.get("endpoint_jobs", [])
            live = [j for j in ep_jobs if self.job_alive(j)]
            if live:
                log.info("[h03_neb] auto_fix: phase 1 — %d endpoint jobs alive", len(live))
                state.set_stage("h03_neb", "RUNNING", job=live[0])
                return True
            # All endpoint jobs dead: if all OUTCARs converged, keep RUNNING → next poll advances
            paths = hs.get("paths", [])
            n_images = hs.get("n_images", _N_IMAGES_DEFAULT)
            n_end = n_images + 1
            if paths:
                all_conv = all(
                    self._outcar_converged(neb_dir / pname / "00" / "OUTCAR") and
                    self._outcar_converged(neb_dir / pname / f"{n_end:02d}" / "OUTCAR")
                    for pname in paths
                )
                if all_conv:
                    log.info("[h03_neb] auto_fix: all endpoints converged — keeping RUNNING for phase 2")
                    state.set_stage("h03_neb", "RUNNING")
                    return True
            log.warning("[h03_neb] auto_fix: endpoints dead/missing — reset to PENDING")
            state.set_stage("h03_neb", "PENDING")
            state.set_handler("h03_neb", {})
            return True

        return False

    # ── Main submit / poll (3 phases, starting from dft/opt/CONTCAR) ────────

    def submit(self, project_dir: Path, state: "ProjectState") -> str | None:
        """Dispatch to the current NEB phase (1–3) and return the submitted SLURM job ID."""
        yaml_data = self.read_project_yaml(project_dir)
        neb_dir   = project_dir / "neb"
        neb_dir.mkdir(parents=True, exist_ok=True)

        hs    = state.get_handler("h03_neb") or {}
        phase = hs.get("phase", 1)
        if phase == 1:
            return self._phase1_setup_and_endpoints(project_dir, yaml_data, neb_dir, state)
        if phase == 2:
            return self._phase2_images(project_dir, yaml_data, neb_dir, state)
        if phase == 3:
            return self._phase3_extract(project_dir, neb_dir, state)
        return None

    def check_progress(self, project_dir: Path, state: "ProjectState") -> None:
        """Poll the current NEB phase progress and advance to the next phase when ready."""
        hs       = state.get_handler("h03_neb") or {}
        neb_dir  = project_dir / "neb"

        phase = hs.get("phase", 1)
        if phase == 1:
            self._check_phase1(project_dir, neb_dir, state, hs)
        elif phase == 2:
            self._check_phase2(project_dir, neb_dir, state, hs)
        elif phase == 3:
            self._phase3_extract(project_dir, neb_dir, state)

    # ── Phase 1: path setup + endpoint optimisation ──────────────────────────

    def _phase1_setup_and_endpoints(self, project_dir: Path, yaml_data: dict,
                                     neb_dir: Path, state: "ProjectState") -> str | None:
        """Set up migration path directories, write endpoint VASP inputs, and submit endpoint jobs."""
        contcar = dft_opt(project_dir) / "CONTCAR"
        n_images  = int(yaml_data.get("neb_images", _N_IMAGES_DEFAULT))
        mobile    = yaml_data.get("mobile_ion", "Li")

        paths = self._find_migration_paths(contcar, mobile, yaml_data)
        if not paths:
            log.error("[h03_neb] No migration paths found in %s", contcar)
            state.set_stage("h03_neb", "FAILED", error="no paths found")
            return None

        log.info("[h03_neb] Found %d migration paths: %s",
                 len(paths), [p["name"] for p in paths])

        potcar = self._make_potcar(project_dir, yaml_data, contcar)
        if potcar is None:
            log.error("[h03_neb] Cannot build POTCAR — check potcar_map in project.yaml")
            state.set_stage("h03_neb", "FAILED", error="POTCAR missing")
            return None

        # ENCUT: explicit yaml override > 1.3 × max(POTCAR ENMAX)
        encut = int(self.project_encut(project_dir, yaml_data))
        log.info("[h03_neb] NEB ENCUT=%d eV",
                 encut)

        kpoints_str = self._make_kpoints(yaml_data, poscar=contcar)
        neb_extra   = yaml_data.get("neb_incar") or None
        incar_ep    = _build_incar("neb_endpoint", encut=encut, extra=neb_extra)
        incar_img   = _build_incar("neb_images",   encut=encut, extra=neb_extra)

        endpoint_jobs = []
        for p in paths:
            pdir = neb_dir / p["name"]
            pdir.mkdir(exist_ok=True)

            # Write i.vasp and f.vasp
            p["i_struct"].to(fmt="poscar", filename=str(pdir / "i.vasp"))
            p["f_struct"].to(fmt="poscar", filename=str(pdir / "f.vasp"))

            # Endpoint dirs 00 and N+1
            n_end = n_images + 1
            for ep_tag, ep_src in (("00", "i.vasp"), (f"{n_end:02d}", "f.vasp")):
                ep_dir = pdir / ep_tag
                ep_dir.mkdir(exist_ok=True)
                shutil.copy2(str(pdir / ep_src), str(ep_dir / "POSCAR"))
                _write_incar(ep_dir / "INCAR", incar_ep)
                (ep_dir / "KPOINTS").write_text(kpoints_str)
                shutil.copy2(str(potcar), str(ep_dir / "POTCAR"))

            # Image INCAR/KPOINTS/POTCAR (written once; copied per-image after nebmake.pl)
            _write_incar(pdir / "INCAR", incar_img)
            (pdir / "KPOINTS").write_text(kpoints_str)
            shutil.copy2(str(potcar), str(pdir / "POTCAR"))

            # sub_endpoints.sh: 2 nodes, each optimises one endpoint
            n_end_tag = f"{n_end:02d}"
            sub_ep = self._write_sub_endpoints(pdir, n_end_tag, yaml_data)
            jid = self._sbatch(sub_ep, cwd=pdir)
            if jid:
                endpoint_jobs.append(jid)
                log.info("[h03_neb] %s: endpoint job %s", p["name"], jid)

        state.set_handler("h03_neb", {
            "phase": 1,
            "paths": [p["name"] for p in paths],
            "endpoint_jobs": endpoint_jobs,
            "n_images": n_images,
        })
        state.set_stage("h03_neb", "RUNNING")
        return endpoint_jobs[0] if endpoint_jobs else None

    def _check_phase1(self, project_dir: Path, neb_dir: Path,
                      state: "ProjectState", hs: dict) -> None:
        """Poll endpoint OUTCARs; advance to phase 2 (image generation) when all converged."""
        paths    = hs.get("paths", [])
        n_images = hs.get("n_images", _N_IMAGES_DEFAULT)
        n_end    = n_images + 1

        # Detect dead endpoint jobs (srun failure, OOM, walltime) — reset to PENDING.
        ep_jobs = hs.get("endpoint_jobs", [])
        if ep_jobs and not any(self.job_alive(j) for j in ep_jobs):
            any_missing = any(
                not (neb_dir / pname / "00" / "OUTCAR").exists()
                or not (neb_dir / pname / f"{n_end:02d}" / "OUTCAR").exists()
                for pname in paths
            )
            if any_missing:
                log.warning("[h03_neb] Endpoint jobs dead but OUTCARs missing — resetting to PENDING")
                state.set_stage("h03_neb", "PENDING")
                state.set_handler("h03_neb", {})
                return

        all_done = True
        for pname in paths:
            pdir = neb_dir / pname
            ep00  = pdir / "00" / "OUTCAR"
            ep_f  = pdir / f"{n_end:02d}" / "OUTCAR"
            if not (ep00.exists() and ep_f.exists()):
                all_done = False
                continue
            # Check convergence (last line of OUTCAR has "reached required accuracy")
            if not self._outcar_converged(ep00) or not self._outcar_converged(ep_f):
                all_done = False

        if all_done:
            log.info("[h03_neb] All endpoints converged — advancing to phase 2")
            state.set_handler("h03_neb", {**hs, "phase": 2})
            self._phase2_images(project_dir, self.read_project_yaml(project_dir),
                                neb_dir, state)

    # ── Phase 2: nebmake.pl + fix_atoms.py + submit image jobs ───────────────

    def _phase2_images(self, project_dir: Path, yaml_data: dict,
                       neb_dir: Path, state: "ProjectState") -> str | None:
        """Interpolate images via nebmake, apply selective dynamics, and submit image jobs."""
        hs       = state.get_handler("h03_neb") or {}
        paths    = hs.get("paths", [])
        n_images = hs.get("n_images", _N_IMAGES_DEFAULT)
        n_end    = n_images + 1
        mobile   = yaml_data.get("mobile_ion", "Li")

        # Dedup guard: don't resubmit if image jobs are already alive
        existing = hs.get("image_jobs", [])
        if existing and any(self.job_alive(j) for j in existing):
            log.info("[h03_neb] phase 2 image jobs already alive — skipping re-submission")
            return existing[0]

        all_image_jobs = []
        for pname in paths:
            pdir  = neb_dir / pname
            ep00  = pdir / "00" / "CONTCAR"
            ep_f  = pdir / f"{n_end:02d}" / "CONTCAR"
            if not ep00.exists():
                ep00 = pdir / "00" / "POSCAR"
            if not ep_f.exists():
                ep_f = pdir / f"{n_end:02d}" / "POSCAR"

            # Run nebmake.pl to generate image directories
            self._run_nebmake(pdir, ep00, ep_f, n_images)

            # Find migrating atom index (1-based) = largest displacement i→f
            mig_idx = self._find_migrating_atom(pdir / "i.vasp", pdir / "f.vasp", mobile)

            # Apply fix_atoms.py selective dynamics to each image
            for img in range(1, n_images + 1):
                img_dir = pdir / f"{img:02d}"
                if not img_dir.exists():
                    continue
                self._apply_selective_dynamics(img_dir, mig_idx)
                # Copy INCAR/KPOINTS/POTCAR
                for f in ("INCAR", "KPOINTS", "POTCAR"):
                    src = pdir / f
                    if src.exists() and not (img_dir / f).exists():
                        shutil.copy2(str(src), str(img_dir / f))

            # Split images across 2 nodes
            imgs_node1 = [f"{i:02d}" for i in range(1, _IMAGES_NODE1 + 1)]
            imgs_node2 = [f"{i:02d}" for i in range(_IMAGES_NODE1 + 1, n_images + 1)]

            sub1 = self._write_sub_images(pdir, "sub_images_1.sh",
                                           imgs_node1, pname, 1, yaml_data)
            jid = self._sbatch(sub1, cwd=pdir)
            if jid:
                all_image_jobs.append(jid)
                log.info("[h03_neb] %s: image job %s (node 1)", pname, jid)
            if imgs_node2:
                sub2 = self._write_sub_images(pdir, "sub_images_2.sh",
                                               imgs_node2, pname, 2, yaml_data)
                jid = self._sbatch(sub2, cwd=pdir)
                if jid:
                    all_image_jobs.append(jid)
                    log.info("[h03_neb] %s: image job %s (node 2)", pname, jid)

        state.set_handler("h03_neb", {**hs, "phase": 2, "image_jobs": all_image_jobs})
        # Update tracked job so the orchestrator's dead-job check sees the new alive job
        if all_image_jobs:
            state.set_stage("h03_neb", "RUNNING", job=all_image_jobs[0])
        return all_image_jobs[0] if all_image_jobs else None

    def _check_phase2(self, project_dir: Path, neb_dir: Path,
                      state: "ProjectState", hs: dict) -> None:
        """Poll image OUTCARs; advance to phase 3 (barrier extraction) when all converged."""
        paths    = hs.get("paths", [])
        n_images = hs.get("n_images", _N_IMAGES_DEFAULT)

        all_done = True
        for pname in paths:
            pdir = neb_dir / pname
            for img in range(1, n_images + 1):
                outcar = pdir / f"{img:02d}" / "OUTCAR"
                if not outcar.exists() or not self._outcar_converged(outcar):
                    all_done = False
                    break

        if all_done:
            log.info("[h03_neb] All images converged — extracting barriers")
            state.set_handler("h03_neb", {**hs, "phase": 3})
            self._phase3_extract(project_dir, neb_dir, state)

    # ── Phase 3: extract barriers ────────────────────────────────────────────

    def _phase3_extract(self, project_dir: Path,
                        neb_dir: Path, state: "ProjectState") -> str | None:
        """Read per-image energies, compute activation barriers, and write pipeline_state.json."""
        hs       = state.get_handler("h03_neb") or {}
        paths    = hs.get("paths", [])
        n_images = hs.get("n_images", _N_IMAGES_DEFAULT)

        pipeline: dict = {"paths": {}}
        barriers: dict = {}

        for pname in paths:
            pdir   = neb_dir / pname
            n_end  = n_images + 1
            energies = []

            for img in range(0, n_end + 1):
                outcar = pdir / f"{img:02d}" / "OUTCAR"
                e = self._read_last_energy(outcar)
                energies.append(e)

            valid = [e for e in energies if e is not None]
            if len(valid) < 3:
                log.warning("[h03_neb] %s: too few energies — skipping", pname)
                pipeline["paths"][pname] = {"stage": "FAILED", "Ea_meV": None}
                continue

            e0, ef = energies[0], energies[-1]
            if e0 is None or ef is None:
                log.warning("[h03_neb] %s: endpoint energy missing — skipping", pname)
                pipeline["paths"][pname] = {"stage": "FAILED", "Ea_meV": None}
                continue
            e_min = min(e0, ef)
            e_max = max((e for e in energies if e is not None), default=e_min)
            ea_mev = round((e_max - e_min) * 1000, 2)

            log.info("[h03_neb] %s: Ea = %.1f meV", pname, ea_mev)
            pipeline["paths"][pname] = {
                "stage":   "COMPLETE",
                "Ea_meV":  ea_mev,
                "mechanism": "vacancy" if "vacancy" in pname else "interstitial",
                "direction": pname.split("_path")[0].replace("vacancy_", "").replace("interstitial_", ""),
                "energies_eV": [round(e - e0, 4) if e is not None else None
                                for e in energies],
            }
            barriers[pname] = {"Ea_meV": ea_mev}

        (neb_dir / "pipeline_state.json").write_text(json.dumps(pipeline, indent=2))

        results_dir = project_dir / "results"
        results_dir.mkdir(exist_ok=True)
        (results_dir / "neb_barriers.json").write_text(json.dumps(barriers, indent=2))

        log.info("[h03_neb] Wrote barriers: %s", barriers)
        state.set_handler("h03_neb", {**hs, "phase": 3, "barriers": barriers})
        state.set_stage("h03_neb", "COMPLETE")
        return None

    # ── Path finding ─────────────────────────────────────────────────────────

    def _find_migration_paths(self, contcar: Path, mobile: str,
                               yaml_data: dict) -> list[dict]:
        """Return list of {name, i_struct, f_struct} for vacancy + interstitial hops."""
        try:
            _ensure_cladue_env()
            from pymatgen.core import Structure
            struct = Structure.from_file(str(contcar))
        except Exception as exc:
            log.error("[h03_neb] Cannot load CONTCAR: %s", exc)
            return []

        paths = []
        mechs = yaml_data.get("neb_mechanism", "both")

        if mechs in ("vacancy", "both"):
            paths += self._vacancy_paths(struct, mobile)
        if mechs in ("interstitial", "both"):
            paths += self._interstitial_paths(struct, mobile)

        return paths

    def _vacancy_paths(self, struct, mobile: str) -> list[dict]:
        """Find unique vacancy hop pairs by dominant crystallographic direction."""
        import numpy as np
        from pymatgen.core import Structure

        mobile_indices = [i for i, s in enumerate(struct) if s.species_string == mobile]
        if len(mobile_indices) < 2:
            return []

        # Build distance matrix between mobile-ion sites
        frac = struct.frac_coords
        seen_pairs: set[tuple] = set()
        hops: list[dict] = []

        for i in mobile_indices:
            dists = np.array([struct.get_distance(i, j) for j in mobile_indices])
            order = np.argsort(dists)
            for j_rank in order[1:6]:   # check 5 nearest neighbours
                j = mobile_indices[j_rank]
                if dists[j_rank] > _Config.get().hpc("neb_mobile_ion_cutoff_A", 5.5):
                    break
                key = (min(i, j), max(i, j))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                df = frac[j] - frac[i]
                # wrap to [-0.5, 0.5]
                df = df - np.round(df)
                label = self._direction_label(df)
                # Keep only first hop per direction label
                if not any(h["dir_label"] == label for h in hops):
                    hops.append({"i": i, "j": j, "dist": dists[j_rank],
                                 "dir_label": label, "df": df})

        paths = []
        for k, hop in enumerate(hops[:4], 1):   # max 4 vacancy paths
            i, j = hop["i"], hop["j"]
            name = f"vacancy_path_{k}_{hop['dir_label']}"

            # i_struct: perfect structure — Li at i, vacancy at j (remove j)
            i_struct = struct.copy()
            i_struct.remove_sites([j])

            # f_struct: Li at j, vacancy at i (remove i)
            f_struct = struct.copy()
            f_struct.remove_sites([i])

            paths.append({"name": name, "i_struct": i_struct, "f_struct": f_struct})
            log.info("[h03_neb] Vacancy path %s: sites %d↔%d dist=%.2f Å",
                     name, i, j, hop["dist"])

        return paths

    def _interstitial_paths(self, struct, mobile: str) -> list[dict]:
        """Find interstitial sites (empty voids) and generate hop paths."""
        try:
            _ensure_cladue_env()
            from pymatgen.analysis.structure_prediction.substitution_probability import \
                SubstitutionProbability
            from pymatgen.analysis.defects.generators import InterstitialGenerator
            from pymatgen.core import Species

            sp = Species(mobile, 1)
            gen = InterstitialGenerator(struct, sp)
            sites = list(gen)[:4]   # up to 4 interstitial sites
        except Exception:
            log.warning("[h03_neb] Interstitial generator failed — skipping interstitial paths")
            return []

        if len(sites) < 2:
            return []

        paths = []
        for k in range(min(2, len(sites) - 1)):
            site_a = sites[k]
            site_b = sites[k + 1]
            name   = f"interstitial_path_{k + 1}"

            i_struct = struct.copy()
            i_struct.insert(0, mobile, site_a.site.frac_coords)

            f_struct = struct.copy()
            f_struct.insert(0, mobile, site_b.site.frac_coords)

            paths.append({"name": name, "i_struct": i_struct, "f_struct": f_struct})
            log.info("[h03_neb] Interstitial path %s: void↔void", name)

        return paths

    @staticmethod
    def _direction_label(df) -> str:
        """Return 'a', 'b', 'c', 'ab', 'ac', 'bc', or 'abc' for a fractional displacement."""
        import numpy as np
        thr = 0.15
        axes = [abs(df[0]) > thr, abs(df[1]) > thr, abs(df[2]) > thr]
        labels = [l for l, active in zip(("a", "b", "c"), axes) if active]
        return "".join(labels) or "diag"

    # ── VASP input writers ────────────────────────────────────────────────────

    @staticmethod
    def _make_kpoints(yaml_data: dict, poscar: "Path | None" = None) -> str:
        """Write Monkhorst-Pack KPOINTS content.

        Priority: explicit neb_kpoints > kpoints_from_poscar (core module) > 1 1 1.
        """
        if "neb_kpoints" in yaml_data:
            mesh = yaml_data["neb_kpoints"]
            if isinstance(mesh, (list, tuple)):
                mesh_str = " ".join(str(x) for x in mesh)
            else:
                mesh_str = str(mesh)
            return f"Automatic\n0\nGamma\n{mesh_str}\n0 0 0\n"

        if poscar is not None and poscar.exists():
            try:
                ka, kb, kc = _kp_from_poscar(poscar)
                return f"Automatic\n0\nGamma\n{ka} {kb} {kc}\n0 0 0\n"
            except Exception:
                pass

        return "Automatic\n0\nGamma\n1 1 1\n0 0 0\n"

    def _make_potcar(self, project_dir: Path, yaml_data: dict,
                     contcar: Path, dest: Path | None = None) -> Path | None:
        """Concatenate elemental POTCARs in element order from CONTCAR/POSCAR."""
        try:
            elements = _read_poscar_elements(contcar)
            potcar_map = yaml_data.get("potcar_map", {})
            mapped = [potcar_map.get(el, el) for el in elements]
        except Exception as exc:
            log.error("[h03_neb] Cannot read elements from %s: %s", contcar, exc)
            return None
        out = dest or (project_dir / "neb" / "POTCAR")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            _build_potcar(mapped, out)
            return out
        except (FileNotFoundError, KeyError) as exc:
            log.error("[h03_neb] Cannot build POTCAR: %s", exc)
            return None

    # ── SLURM script writers ──────────────────────────────────────────────────

    def _write_sub_endpoints(self, pdir: Path, n_end_tag: str,
                              yaml_data: dict) -> Path:
        """Write SLURM sub_endpoints.sh for the 00 and N+1 endpoint optimisations."""
        job_name = f"neb_ep_{pdir.name[:16]}"
        wall = yaml_data.get("neb_endpoint_wall",
                             _Config.get().slurm_time("neb_endpoint", "48:00:00"))
        out = pdir / "sub_endpoints.sh"
        _write_sub(out, "vasp_neb_endpoints", job_name,
                   endpoint_tags=["00", n_end_tag], time=wall)
        return out

    def _write_sub_images(self, pdir: Path, filename: str, img_tags: list[str],
                           path_name: str, node_idx: int,
                           yaml_data: dict) -> Path:
        """Write SLURM image job script for the given image tags on one node."""
        job_name = f"neb_n{node_idx}_{pdir.name[:12]}"
        wall = yaml_data.get("neb_image_wall",
                             _Config.get().slurm_time("neb_image", "150:00:00"))
        out = pdir / filename
        _write_sub(out, "vasp_neb_images", job_name,
                   image_tags=img_tags, cores_per_image=_CORES_PER_IMAGE,
                   cores_per_node=_CORES_PER_NODE, time=wall)
        return out

    # ── NEB image generation helpers (pure Python, no external scripts) ──────

    def _run_nebmake(self, pdir: Path, i_contcar: Path,
                     f_contcar: Path, n_images: int) -> None:
        """Generate NEB image POSCARs using nonlinear L-BFGS-B path predictor.

        Uses hpca.core.neb.path_finder.build_nonlinear_chained_path (pymatgen-based)
        which curves images away from framework atoms.  Falls back to the pure-numpy
        linear interpolation in hpca.core.neb.linear if pymatgen is unavailable.
        """
        try:
            import numpy as np
            from hpca.core.neb.poscar_io import read_structure, write_poscar
            from hpca.core.neb.path_finder import build_nonlinear_chained_path

            i_struct = read_structure(i_contcar)
            f_struct = read_structure(f_contcar)
            lattice = i_struct.lattice.matrix

            # Identify migrating atom as the one with the largest Cartesian displacement
            diffs = np.linalg.norm(
                i_struct.cart_coords - f_struct.cart_coords, axis=1
            )
            mig_idx = int(np.argmax(diffs))

            start_frac = i_struct.frac_coords[mig_idx]
            end_frac   = f_struct.frac_coords[mig_idx]
            framework_fracs = np.delete(i_struct.frac_coords, mig_idx, axis=0)

            coords = build_nonlinear_chained_path(
                [start_frac, end_frac], framework_fracs, lattice,
                n_images_total=n_images, spacing=0.5,
            )

            if len(coords) != n_images:
                raise ValueError(
                    f"predictor returned {len(coords)} images, expected {n_images}"
                )

            for img_idx, frac in enumerate(coords, start=1):
                img_dir = pdir / f"{img_idx:02d}"
                img_dir.mkdir(exist_ok=True)
                img_struct = i_struct.copy()
                img_struct.replace(mig_idx, i_struct[mig_idx].species_string, frac)
                write_poscar(img_struct, img_dir / "POSCAR")

            log.info("[h03_neb] Nonlinear path: %d images written to %s", n_images, pdir)

        except Exception as exc:
            log.warning("[h03_neb] Nonlinear path predictor failed (%s) — falling back to linear", exc)
            try:
                make_neb_images(i_contcar, f_contcar, n_images, pdir)
            except Exception as exc2:
                log.error("[h03_neb] make_neb_images fallback also failed: %s", exc2)

    def _find_migrating_atom(self, i_vasp: Path, f_vasp: Path,
                              mobile: str) -> int:
        """Return 1-based index of the atom with largest displacement i→f."""
        try:
            return find_migrating_atom(i_vasp, f_vasp, mobile_element=mobile) + 1
        except Exception:
            return 1

    def _apply_selective_dynamics(self, img_dir: Path, mig_idx: int) -> None:
        """Apply constrained selective dynamics to one NEB image POSCAR.

        Uses hpca.core.neb.image_tools.apply_constrained_path_dynamics (pymatgen):
        migrating atom → F F F; atoms within 4 Å → T T T; all others → F F F.
        Falls back to hpca.core.neb.apply_selective_dynamics on failure.

        Parameters
        ----------
        img_dir : Path
            Image directory containing POSCAR.
        mig_idx : int
            1-based index of the migrating atom.
        """
        poscar = img_dir / "POSCAR"
        if not poscar.exists():
            return
        shutil.copy2(str(poscar), str(img_dir / "POSCAR_original"))
        try:
            from hpca.core.neb.poscar_io import read_structure, write_poscar
            from hpca.core.neb.image_tools import apply_constrained_path_dynamics
            struct = read_structure(poscar)
            new_struct = apply_constrained_path_dynamics(struct, mig_idx - 1)  # 0-based
            write_poscar(new_struct, poscar)
        except Exception as exc:
            log.warning("[h03_neb] apply_constrained_path_dynamics failed for %s: %s — falling back",
                        img_dir.name, exc)
            try:
                apply_selective_dynamics(poscar, mobile_indices=[mig_idx - 1], inplace=True)
            except Exception as exc2:
                log.warning("[h03_neb] apply_selective_dynamics fallback also failed: %s", exc2)

    # ── Convergence + energy helpers ─────────────────────────────────────────

    @staticmethod
    def _outcar_converged(outcar: Path) -> bool:
        """Return True if 'reached required accuracy' appears in the last 4000 bytes of OUTCAR."""
        if not outcar.exists():
            return False
        try:
            tail = outcar.read_text()[-4000:]
            return "reached required accuracy" in tail
        except Exception:
            return False

    @staticmethod
    def _read_last_energy(outcar: Path) -> float | None:
        """Return the last 'energy without entropy' value from OUTCAR, or None if absent."""
        if not outcar.exists():
            return None
        try:
            e = None
            for line in outcar.read_text().splitlines():
                if "energy  without entropy" in line:
                    e = float(line.split()[-1])
            return e
        except Exception:
            return None

    # ── sbatch helper ────────────────────────────────────────────────────────

    @staticmethod
    def _sbatch(script: Path, cwd: Path) -> str | None:
        """Submit through the scheduler adapter."""
        from hpca.scheduler import get_scheduler
        return get_scheduler().submit(script, cwd=cwd)

    def on_complete(self, project_dir: Path, state: "ProjectState") -> None:
        """Log the final computed NEB barriers after the handler reaches COMPLETE."""
        hs = state.get_handler("h03_neb") or {}
        barriers = hs.get("barriers", {})
        if barriers:
            log.info("[h03_neb] Final barriers: %s", barriers)
