"""The additive installer must preserve live source files on any conflict."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_v090_test", ROOT / "install_nbs_v090.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def fixture(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(); target.mkdir()
    (source / "old.py").write_text("audited old source\n")
    (source / "new_v090.py").write_text("new runner\n")
    (source / "INSTALL_MANIFEST_v090.json").write_text(json.dumps({
        "dependencies": {"old.py": installer.digest(source / "old.py")},
        "additions": {"new_v090.py": installer.digest(source / "new_v090.py")}}))
    return source, target


def test_conflict_preflight_prevents_any_addition(tmp_path):
    source, target = fixture(tmp_path)
    (target / "old.py").write_text("local active change\n")
    with pytest.raises(FileExistsError, match="preserved without writes"):
        installer.install(source, target)
    assert (target / "old.py").read_text() == "local active change\n"
    assert not (target / "new_v090.py").exists()


def test_dry_run_and_idempotence_preserve_old_file_mtime(tmp_path):
    source, target = fixture(tmp_path)
    (target / "old.py").write_bytes((source / "old.py").read_bytes())
    before = (target / "old.py").stat().st_mtime_ns
    assert installer.install(source, target, dry_run=True)["new_files"] == ["new_v090.py"]
    assert not (target / "new_v090.py").exists()
    installer.install(source, target)
    assert (target / "old.py").stat().st_mtime_ns == before
    second = installer.install(source, target)
    assert second["new_files"] == [] and second["existing_files_rewritten"] == 0
