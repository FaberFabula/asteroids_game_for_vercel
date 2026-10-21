"""Load the bundled Python native extensions in Pygbag's WebAssembly runtime."""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import time
import zipfile

_PREPARED = False
_NATIVE_HANDLES: list[object] = []


async def _wait_for_preload(embed: object, label: str, timeout: float = 45.0) -> None:
    """Yield to JavaScript compilation, with a bounded wait for each batch."""
    deadline = time.monotonic() + timeout
    await asyncio.sleep(0)
    while embed.preloading() > 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out compiling {label}")
        await asyncio.sleep(0)


async def prepare_browser() -> None:
    """Verify, extract and link native wheels before importing the Python game.

    Desktop Python already has normal installed dependencies. The browser must
    compile its bundled native extensions and expose GEOS symbols globally so
    Shapely can link to them. No package downloads occur in this function.
    """
    global _PREPARED
    if sys.platform != "emscripten" or _PREPARED:
        return
    import embed

    vendor = Path(__file__).with_name("vendor")
    packages = json.loads((vendor / "manifest.json").read_text())["packages"]
    # Check every archive before installing any of them. The manifest also
    # records the upstream source and the provenance of each pinned checksum.
    for package in packages:
        wheel = vendor / package["file"]
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != package["sha256"]:
            raise RuntimeError(f"Bundled dependency failed its checksum: {wheel.name}")

    target = Path(sysconfig.get_path("platlib"))
    target.mkdir(parents=True, exist_ok=True)
    for package in packages:
        with zipfile.ZipFile(vendor / package["file"]) as archive:
            archive.extractall(target)
    # Pygbag's sysconfig installation path is not included in its initial path.
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    importlib.invalidate_caches()

    geos_directory = target / "shapely.libs"
    os.environ["LD_LIBRARY_PATH"] = str(geos_directory)
    # Keep path strings alive while the native asynchronous preloader uses them.
    preload_paths: list[str] = []
    geos_libraries = [geos_directory / name for name in ("libgeos.so", "libgeos_c.so")]
    for library in geos_libraries:
        preload_paths.append(str(library))
        embed.preload(preload_paths[-1])
        await _wait_for_preload(embed, library.name)

    # GEOS is already compiled. Compile the Python extensions as one batch.
    for current, directories, files in os.walk(target):
        directories.sort()
        if "shapely.libs" in directories:
            directories.remove("shapely.libs")
        for filename in sorted(files):
            if filename.endswith(".so"):
                preload_paths.append(str(Path(current) / filename))
                embed.preload(preload_paths[-1])
    await _wait_for_preload(embed, "Python native extensions")

    # Compilation alone leaves GEOS local to its library. Shapely needs the
    # public C symbols in the shared process namespace, in dependency order.
    import ctypes
    for library in geos_libraries:
        _NATIVE_HANDLES.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL))

    # Some Pygbag images include a namespace placeholder named pygame. Reimport
    # the complete bundled package instead of retaining that empty namespace.
    placeholder = sys.modules.get("pygame")
    if placeholder is not None and not hasattr(placeholder, "init"):
        for name in list(sys.modules):
            if name == "pygame" or name.startswith("pygame."):
                del sys.modules[name]
    importlib.invalidate_caches()
    _PREPARED = True
