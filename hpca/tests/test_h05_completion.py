"""Regression tests for CMD completion while LAMMPS is still writing."""
from pathlib import Path

from hpca.orchestrator.handlers.h05_cmd import ClassicalMDHandler


def _write_nvt(root: Path, temperature: int, *, finished: bool) -> None:
    nvt = root / "cmd" / "nvt" / str(temperature)
    nvt.mkdir(parents=True)
    (nvt / "dump_unwrapped.lmp").write_bytes(b"trajectory")
    if finished:
        (nvt / "after_nvt_.dat").write_text("final coordinates\n")
        (nvt / "log.lammps").write_text("Loop time ...\nTotal wall time: 1:00:00\n")
    else:
        (nvt / "log.lammps").write_text("500000 300.0 -1234.0\n")


def test_large_growing_dump_is_not_cmd_completion(tmp_path: Path, monkeypatch):
    handler = ClassicalMDHandler()
    monkeypatch.setattr(handler, "nvt_temperatures", lambda: [300])
    monkeypatch.setattr(ClassicalMDHandler, "_min_dump_size", staticmethod(lambda _: 1))
    _write_nvt(tmp_path, 300, finished=False)

    assert not handler.is_complete(tmp_path, None)


def test_final_lammps_artifacts_complete_cmd(tmp_path: Path, monkeypatch):
    handler = ClassicalMDHandler()
    monkeypatch.setattr(handler, "nvt_temperatures", lambda: [300, 400])
    monkeypatch.setattr(ClassicalMDHandler, "_min_dump_size", staticmethod(lambda _: 1))
    _write_nvt(tmp_path, 300, finished=True)
    _write_nvt(tmp_path, 400, finished=True)

    assert handler.is_complete(tmp_path, None)


def test_nvt_finalizer_rejects_partial_large_output(tmp_path: Path):
    _write_nvt(tmp_path, 300, finished=False)
    nvt = tmp_path / "cmd" / "nvt" / "300"

    assert not ClassicalMDHandler._nvt_run_complete(nvt, 1)

    (nvt / "after_nvt_.dat").write_text("final coordinates\n")
    (nvt / "log.lammps").write_text("Total wall time: 0:10:00\n")
    assert ClassicalMDHandler._nvt_run_complete(nvt, 1)
