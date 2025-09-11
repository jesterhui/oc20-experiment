import lzma
from pathlib import Path

from oc20_exp.data.s2ef.io import load_metadata, find_data_files


def test_load_metadata_tmpfile(tmp_path: Path):
    txt_path = tmp_path / "0.txt.xz"
    content = "1,0,-1.23\n2,1,-2.34\n"
    with lzma.open(txt_path, "wt") as f:
        f.write(content)

    metas = load_metadata(txt_path)
    assert len(metas) == 2
    assert metas[0].system_id == "1"
    assert metas[1].reference_energy == -2.34


def test_find_data_files_pairs(tmp_path: Path):
    # Create matching pair 0.extxyz.xz <-> 0.txt.xz
    (tmp_path / "0.extxyz.xz").write_bytes(b"")
    with lzma.open(tmp_path / "0.txt.xz", "wt") as f:
        f.write("1,0,-1.0\n")

    # Non-matching extxyz without txt should be ignored
    (tmp_path / "1.extxyz.xz").write_bytes(b"")

    pairs = find_data_files(tmp_path, logger=_DummyLogger())
    assert len(pairs) == 1
    assert pairs[0][0].name == "0.extxyz.xz"
    assert pairs[0][1].name == "0.txt.xz"


class _DummyLogger:
    def warning(self, *args, **kwargs):
        pass
