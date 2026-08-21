"""NEB submission generation belongs exclusively to the submission registry."""
from pathlib import Path

import pytest

from hpca.registry.submission import write_submission


def test_neb_fanout_is_local_launcher(tmp_path: Path) -> None:
    path = write_submission(
        tmp_path / "submit_all.sh", "submit_fanout", "neb_all",
        scripts=["vacancy/sub_endpoints.sh", "vacancy/sub_images_1.sh"],
    )
    text = path.read_text()
    assert "#SBATCH" not in text
    assert "sbatch vacancy/sub_endpoints.sh" in text
    assert "sbatch vacancy/sub_images_1.sh" in text
    assert path.stat().st_mode & 0o111


def test_neb_fanout_rejects_empty_children(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        write_submission(tmp_path / "submit_all.sh", "submit_fanout", "empty", scripts=[])


def test_neb_domain_contains_no_scheduler_templates() -> None:
    neb_root = Path(__file__).resolve().parents[1] / "core/neb"
    for source in neb_root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "#SBATCH" not in text
        assert "srun " not in text
