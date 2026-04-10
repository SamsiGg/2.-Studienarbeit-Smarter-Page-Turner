# =============================================================================
# omr.py – PDF → MusicXML via Audiveris (OMR)
# =============================================================================
# Wrapper um Audiveris CLI. Kostenlos, unbegrenzt, Open Source.
#
# Installation (macOS):
#   Audiveris.app herunterladen und in /Applications verschieben:
#   https://github.com/Audiveris/audiveris/releases
#
# Das Skript findet Audiveris automatisch unter:
#   1. /Applications/Audiveris.app
#   2. AUDIVERIS_PATH Env-Variable
#   3. 'audiveris' im PATH
# =============================================================================

import os
import subprocess
from pathlib import Path

# Typische Installationspfade (macOS)
_MAC_APP_EXECUTABLE = "/Applications/Audiveris.app/Contents/MacOS/Audiveris"


_GENERATED_DIR = Path(__file__).parent.parent.parent / "data" / "generated"


def convert_pdf(pdf_path: str, output_dir: str | None = None) -> str:
    """Konvertiert eine PDF-Datei zu MusicXML via Audiveris OMR.

    Args:
        pdf_path: Pfad zur PDF-Datei.
        output_dir: Ausgabe-Ordner (Standard: neben der PDF).

    Returns:
        Pfad zur erzeugten .mxl Datei.

    Raises:
        FileNotFoundError: Wenn PDF oder Audiveris nicht gefunden.
        RuntimeError: Wenn Audiveris fehlschlägt.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")

    if output_dir is None:
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(_GENERATED_DIR)

    # Audiveris-Executable finden
    audiveris_bin = _find_audiveris()
    if audiveris_bin is None:
        _print_install_help()
        raise FileNotFoundError("Audiveris nicht gefunden. Siehe Installationshinweise oben.")

    cmd = [
        audiveris_bin,
        "-batch", "-export",
        "-output", output_dir,
        "--", str(pdf_path),
    ]

    print(f"Starte Audiveris OMR: {pdf_path.name}")
    print(f"  Executable: {audiveris_bin}")

    # Snapshot vor dem Aufruf: bereits vorhandene .mxl-Dateien merken
    existing_mxl = set(Path(output_dir).rglob("*.mxl")) if Path(output_dir).exists() else set()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  Audiveris stderr: {result.stderr[:500]}")
        raise RuntimeError(f"Audiveris Fehler (Code {result.returncode})")

    # Ausgabe-Datei finden (.mxl) – zuerst nach dem Stem, dann nur neue Dateien
    mxl_path = _find_output_mxl(output_dir, pdf_path.stem, existing_mxl)
    if mxl_path is None:
        raise RuntimeError(
            f"Audiveris hat keine .mxl Datei erzeugt. "
            f"Prüfe {output_dir} manuell."
        )

    print(f"  MusicXML erzeugt: {mxl_path}")
    return mxl_path


def _find_audiveris() -> str | None:
    """Sucht die Audiveris-Executable."""
    # 1. Env-Variable (höchste Priorität)
    env_path = os.environ.get("AUDIVERIS_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. macOS App-Bundle
    if Path(_MAC_APP_EXECUTABLE).exists():
        return _MAC_APP_EXECUTABLE

    # 3. Im PATH (Linux, oder manuell installiert)
    try:
        result = subprocess.run(
            ["which", "audiveris"], capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass

    return None


def _find_output_mxl(output_dir: str, stem: str, existing_mxl: set | None = None) -> str | None:
    """Sucht die erzeugte .mxl Datei in typischen Audiveris-Ausgabepfaden."""
    output_path = Path(output_dir)

    # Direkte Kandidaten (exakter Stem)
    candidates = [
        output_path / f"{stem}.mxl",
        output_path / stem / f"{stem}.mxl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Audiveris hängt manchmal ".mvt1" o.ä. an – nach Stem-Präfix suchen
    if output_path.exists():
        for mxl in output_path.rglob("*.mxl"):
            if mxl.name.startswith(stem):
                return str(mxl)

    # Letzter Fallback: nur Dateien, die nach dem Audiveris-Aufruf neu entstanden sind
    if existing_mxl is not None and output_path.exists():
        new_files = set(output_path.rglob("*.mxl")) - existing_mxl
        if new_files:
            return str(next(iter(new_files)))

    return None


def _print_install_help():
    """Gibt Installationshinweise für Audiveris aus."""
    print("\n" + "=" * 60)
    print("  AUDIVERIS NICHT GEFUNDEN")
    print("=" * 60)
    print()
    print("  Audiveris wird fuer PDF -> MusicXML benoetigt.")
    print("  Fuer MusicXML-Input ist Audiveris NICHT noetig.")
    print()
    print("  Installation (macOS):")
    print("    1. Herunterladen:")
    print("       https://github.com/Audiveris/audiveris/releases")
    print("    2. Audiveris.app nach /Applications verschieben")
    print()
    print("  Alternativ: AUDIVERIS_PATH Env-Variable setzen:")
    print('    export AUDIVERIS_PATH="/pfad/zu/Audiveris"')
    print()
    print("=" * 60 + "\n")
