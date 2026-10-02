import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from core.errors import AppError, ErrorCode
from core.parsing import DOC, parse_file
from tests.builders import make_docx

SOFFICE = shutil.which("soffice") or (
    "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    if Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").is_file()
    else None
)


@pytest.mark.skipif(SOFFICE is None, reason="LibreOffice isn't installed")
def test_real_legacy_doc_converts_and_parses() -> None:
    assert SOFFICE is not None
    with tempfile.TemporaryDirectory(prefix="docmind-doc-test-") as directory:
        root = Path(directory)
        source = root / "policy.docx"
        source.write_bytes(make_docx())
        converted = subprocess.run(
            [
                SOFFICE,
                "--headless",
                "--nologo",
                "--nodefault",
                "--nofirststartwizard",
                "--convert-to",
                "doc",
                "--outdir",
                directory,
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=45,
        )
        legacy_file = root / "policy.doc"
        assert converted.returncode == 0, converted.stderr
        assert legacy_file.is_file()

        parsed = parse_file(legacy_file.read_bytes(), DOC, libreoffice_path=SOFFICE)

    assert parsed.pages[0].text.startswith("# Leave policy")
    assert "Employees get 25 days of paid leave." in parsed.pages[0].text


def test_missing_converter_returns_clear_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.doc_conversion.shutil.which", lambda _: None)
    monkeypatch.setattr("core.doc_conversion._MACOS_SOFFICE", Path("/missing/soffice"))

    with pytest.raises(AppError, match="LibreOffice") as exc:
        parse_file(b"legacy word bytes", DOC)

    assert exc.value.code is ErrorCode.PARSE_FAILED
