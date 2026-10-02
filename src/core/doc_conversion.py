import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core.errors import AppError, ErrorCode
from core.logging import get_logger

log = get_logger(__name__)
CONVERSION_TIMEOUT_SECONDS = 45
_MACOS_SOFFICE = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")


def convert_doc_to_docx(data: bytes, converter_path: str | None = None) -> bytes:
    executable = _find_converter(converter_path)
    if executable is None:
        raise AppError(
            ErrorCode.PARSE_FAILED,
            "Legacy .doc files require LibreOffice. Install LibreOffice or configure LIBREOFFICE_PATH.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="docmind-doc-") as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "converted"
            profile_dir = root / "profile"
            source_dir.mkdir()
            output_dir.mkdir()
            profile_dir.mkdir()
            source_path = source_dir / "document.doc"
            output_path = output_dir / "document.docx"
            source_path.write_bytes(data)

            result = subprocess.run(
                [
                    executable,
                    f"-env:UserInstallation={profile_dir.as_uri()}",
                    "--headless",
                    "--nologo",
                    "--nodefault",
                    "--nofirststartwizard",
                    "--nolockcheck",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    os.fspath(output_dir),
                    os.fspath(source_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=CONVERSION_TIMEOUT_SECONDS,
            )
            if result.returncode != 0 or not output_path.is_file():
                log.warning("legacy_doc_conversion_failed", return_code=result.returncode)
                raise AppError(ErrorCode.PARSE_FAILED, "This Word 97-2003 document could not be converted.")
            return output_path.read_bytes()
    except subprocess.TimeoutExpired:
        raise AppError(ErrorCode.PARSE_FAILED, "This Word document took too long to convert.") from None
    except OSError as exc:
        log.warning("legacy_doc_conversion_failed", error_type=type(exc).__name__)
        raise AppError(ErrorCode.PARSE_FAILED, "This Word 97-2003 document could not be converted.") from None


def _find_converter(configured_path: str | None) -> str | None:
    if configured_path:
        candidate = shutil.which(configured_path)
        if candidate:
            return candidate
        path = Path(configured_path)
        if path.is_file() and os.access(path, os.X_OK):
            return os.fspath(path)
        return None
    return (
        shutil.which("soffice")
        or shutil.which("libreoffice")
        or (os.fspath(_MACOS_SOFFICE) if _MACOS_SOFFICE.is_file() else None)
    )
