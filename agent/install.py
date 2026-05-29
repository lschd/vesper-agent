#!/usr/bin/env python3
"""Installer autonomo per Vesper."""
import os
import subprocess
import sys
from pathlib import Path

# ── 0. Bootstrap venv (solo stdlib — nessun import di terze parti prima di qui) ──

_ROOT = Path(__file__).resolve().parent

_MIN_PY = (3, 10)
_PREFERRED_PY_VERSIONS = [(3, 13), (3, 12), (3, 11), (3, 10)]


def _find_preferred_python() -> list[str]:
    """Restituisce il comando per la versione Python più recente >= _MIN_PY trovata.

    Su Windows prova il py launcher (py -3.13, -3.12, …). Su Linux prova python3.13, ecc.
    Esce con errore se nessuna versione >= _MIN_PY è trovata.
    """
    for major, minor in _PREFERRED_PY_VERSIONS:
        ver_str = f"{major}.{minor}"
        cmd = ["py", f"-{ver_str}"] if sys.platform == "win32" else [f"python{ver_str}"]
        try:
            r = subprocess.run(cmd + ["--version"], capture_output=True, timeout=5)
            if r.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    sys.exit(
        f"\n❌ Python {_MIN_PY[0]}.{_MIN_PY[1]}+ non trovato.\n"
        f"   Installalo da https://python.org/downloads/ e riesegui install.py."
    )

def _fail(msg: str, code: int = 1) -> None:
    print(f"\n❌ {msg}", file=sys.stderr)
    sys.exit(code)


def step(label: str) -> None:
    print(f"\n▶ {label}")

if sys.platform == "win32":
    _venv_python = _ROOT / ".venv" / "Scripts" / "python.exe"
else:
    _venv_python = _ROOT / ".venv" / "bin" / "python"

if not _venv_python.exists():
    step("Creazione venv...")
    _py_cmd = _find_preferred_python()
    subprocess.run(_py_cmd + ["-m", "venv", str(_ROOT / ".venv")], check=True)
    print(f"  venv creato ✓")

if Path(sys.prefix).resolve() != (_ROOT / ".venv").resolve():
    print("  Ri-esecuzione con Python del venv...")
    if sys.platform == "win32":
        try:
            sys.exit(subprocess.run([str(_venv_python)] + sys.argv).returncode)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

# Dentro il venv: blocca se la versione Python è inferiore al minimo.
if sys.version_info[:2] < _MIN_PY:
    sys.exit(
        f"\n❌ Il venv usa Python {sys.version.split()[0]}, ma serve >= "
        f"{_MIN_PY[0]}.{_MIN_PY[1]}.\n"
        f"   Ricrea il venv con una versione supportata:\n"
        f"     Remove-Item -Recurse -Force .venv  (Windows)\n"
        f"     rm -rf .venv                       (Linux/WSL)\n"
        f"     python install.py"
    )

# Da qui in poi siamo sicuramente dentro il venv — import di terze parti consentiti.

step("Installo uv...")
subprocess.run([str(_venv_python), "-m", "pip", "install", "-q", "python-dotenv"], check=True)
print(f"  uv installato ✓")

import shutil  # noqa: E402
import stat  # noqa: E402

try:
    from dotenv import load_dotenv
except ImportError:
    print("  Installazione python-dotenv...")
    # subprocess.run([str(_venv_python), "-m", "pip", "install", "-q", "python-dotenv"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "python-dotenv"], check=True)
    from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_REPO = os.getenv("LLM_MODEL_REPO", "unsloth/Qwen3.6-27B-GGUF")
LLM_MODEL_FILE = os.getenv("LLM_MODEL_FILE", "Qwen3.6-27B-Q6_K.gguf")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL_REPO = os.getenv("RERANKER_MODEL_REPO", "Mungert/Qwen3-Reranker-4B-GGUF")
RERANKER_MODEL_FILE = os.getenv("RERANKER_MODEL_FILE", "Qwen3-Reranker-4B-Q4_K_M.gguf")
LLM_MANAGED = os.getenv("LLM_MANAGED", "true").lower() == "true"

ROOT = _ROOT


def _read_raw_env_value(key: str) -> str:
    """
    Legge un valore dal .env senza interpretare escape sequences.

    python-dotenv processa i valori quotati come stringhe Python:
    C:\\Users\\nome\\agent → C:\\Users\\nome\\<chr7>gent (\\a = bell).
    Questa funzione legge il file direttamente, restituendo il testo letterale.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() != key:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
            v = v[1:-1]
        return v
    return ""


def _abs_dir(env_key: str, default: Path) -> Path:
    # Su Windows leggi sempre raw dal .env: python-dotenv interpreta i backslash
    # dei path Windows come escape Python (\agent → chr(7)gent, \temp → tab+emp…).
    # Se il valore non è nel .env (variabile d'ambiente di sistema), usa os.getenv
    # che restituisce la stringa letterale senza elaborazione.
    if sys.platform == "win32":
        val = _read_raw_env_value(env_key) or os.getenv(env_key, "")
    else:
        val = os.getenv(env_key, "")

    if not val:
        return default

    # Converti path WSL /mnt/X/... → X:/...
    if sys.platform == "win32" and len(val) >= 7 and val[:5] == "/mnt/" and val[6:7] == "/":
        val = f"{val[5].upper()}:{val[6:]}"

    p = Path(val)
    return (ROOT / p) if not p.is_absolute() else p


LLM_MODEL_DIR = ROOT / "data" / "models" / Path(LLM_MODEL_REPO)
EMBEDDING_LOCAL_DIR = ROOT / "data" / "models" / Path(EMBEDDING_MODEL)
RERANKER_MODEL_DIR = ROOT / "data" / "models" / Path(RERANKER_MODEL_REPO)
VAULT_PATH = _abs_dir("VAULT_PATH", ROOT / "vault")

DIRS_TO_CREATE = [
    ROOT / "data" / "models",
    ROOT / "data" / "chroma",
    ROOT / "data" / "logs",
    VAULT_PATH / "raw",
    VAULT_PATH / "wiki",
    VAULT_PATH / "synthesis",
    VAULT_PATH / "agenda" / "recurring",
    VAULT_PATH / "agenda" / "one-off",
]

def _download_llama_server_windows() -> bool:
    """Scarica llama-server.exe precompilato con CUDA da llama.cpp GitHub releases.

    Il binario include le CUDA DLL bundled (cudart, cublas, …): non richiede
    né CUDA Toolkit né Visual C++ Build Tools installati sul sistema.
    """
    import json
    import urllib.request
    import zipfile

    bin_dir = ROOT / "data" / "bin" / "llama-server"
    server_exe = bin_dir / "llama-server.exe"

    if server_exe.exists():
        print("  llama-server.exe già presente ✓")
        return True

    print("  Recupero ultima release da GitHub (ggml-org/llama.cpp)...")
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "vesper-installer"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = json.loads(resp.read())
    except Exception as e:
        print(f"  ⚠️  GitHub API non raggiungibile: {e}\n  Avvia llama-server manualmente o imposta LLM_MANAGED=false.")
        return False

    tag = release.get("tag_name", "?")
    assets = release.get("assets", [])

    # Asset Windows CUDA x64: cerca preferendo versioni CUDA più recenti
    cuda_asset = None
    for a in sorted(assets, key=lambda x: x["name"], reverse=True):
        n = a["name"].lower()
        if "win" in n and "cuda" in n and "x64" in n and n.endswith(".zip"):
            cuda_asset = a
            break

    if not cuda_asset:
        names = [a["name"] for a in assets]
        print(f"  ⚠️  Nessun asset Windows CUDA trovato nella release {tag}.\n  Asset disponibili: {names}")
        return False

    size_mb = cuda_asset["size"] // (1024 * 1024)
    print(f"  Release {tag} — download {cuda_asset['name']} ({size_mb} MB)...")
    bin_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bin_dir / cuda_asset["name"]

    def _progress(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            done = count * block // (1024 * 1024)
            print(f"\r  [{pct:3d}%] {done}/{total // (1024*1024)} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(cuda_asset["browser_download_url"], zip_path, _progress)
        print()
    except Exception as e:
        print(f"\n  ⚠️  Download fallito: {e}")
        return False

    print("  Estrazione...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            fname = Path(member.filename).name
            if fname and not member.is_dir() and (fname.endswith(".exe") or fname.endswith(".dll")):
                (bin_dir / fname).write_bytes(zf.read(member))

    zip_path.unlink(missing_ok=True)

    if server_exe.exists():
        print(f"  llama-server.exe installato ✓  (CUDA, {tag})")
        return True

    print("  ⚠️  llama-server.exe non trovato dopo l'estrazione.")
    return False


def _detect_cuda_tag() -> str | None:
    """Rileva la versione CUDA installata e restituisce il tag wheel (es. 'cu124')."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            if "CUDA Version:" in line:
                version = line.split("CUDA Version:")[-1].strip().split()[0]
                major, minor = version.split(".")[:2]
                return f"cu{major}{int(minor)}"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _get_cuda_lib_dir_linux() -> str | None:
    """Trova la directory CUDA runtime su Linux/WSL (libcudart.so*)."""
    import glob as _glob
    _CANDIDATES = [
        "/usr/local/cuda/lib64",
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/lib/wsl/lib",
        "/usr/lib/x86_64-linux-gnu",
    ]
    for d in _CANDIDATES:
        if _glob.glob(f"{d}/libcudart.so*"):
            return d
    return None


def _check_binary_cuda_linux(bin_path: Path) -> tuple[bool, bool]:
    """Controlla se il binario è compilato con CUDA e se le librerie sono raggiungibili.

    Ritorna (cuda_linked, cuda_found):
      cuda_linked — True se il binario ha dipendenze CUDA
      cuda_found  — True se libcudart/libcublas sono risolte a runtime
    """
    cuda_dir = _get_cuda_lib_dir_linux()
    env = os.environ.copy()
    ld_parts = [str(bin_path.parent)]
    if cuda_dir:
        ld_parts.insert(0, cuda_dir)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts + ([existing] if existing else []))

    try:
        r = subprocess.run(
            ["ldd", str(bin_path)], capture_output=True, text=True, env=env, timeout=10
        )
        out = r.stdout
        if not out:
            raise FileNotFoundError  # ldd non disponibile o output vuoto
        out_lower = out.lower()
        if "cuda" not in out_lower and "cublas" not in out_lower:
            return False, False  # binario CPU-only
        cuda_missing = any(
            "not found" in line
            for line in out.splitlines()
            if "cuda" in line.lower() or "cublas" in line.lower()
        )
        return True, not cuda_missing
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ldd non disponibile — test diretto con --version
        try:
            r2 = subprocess.run(
                [str(bin_path), "--version"],
                capture_output=True, text=True, env=env, timeout=15,
            )
            return True, r2.returncode == 0 and cuda_dir is not None
        except (OSError, subprocess.TimeoutExpired):
            return False, False


def _llama_has_cuda_backend() -> bool:
    """Verifica che llama-cpp-python nel venv abbia backend CUDA compilato.

    Pattern in ordine:
    1a. dist-info tag: wheel vecchio formato con +cu1 nel nome (es. llama_cpp_python-0.3.9+cu124)
    1b. direct_url.json: installato da URL diretto con tag CUDA nel path (formato 0.3.20+)
    2.  libggml-cuda*: librerie separate presenti nel package (layout 0.2.x / 0.3.x early)
    3.  ldd su .so nel package: CUDA linkato dinamicamente (layout 0.3.x+)
    """
    import glob as _glob
    import json as _json
    lib_dir = ROOT / ".venv" / "lib"

    # Pattern 1a: dist-info tag wheel vecchio formato (llama_cpp_python-0.3.9+cu124.dist-info)
    if _glob.glob(str(lib_dir / "**" / "llama_cpp_python-*cu1*.dist-info"), recursive=True):
        return True

    # Pattern 1b: direct_url.json contiene tag CUDA nel path dell'URL (formato 0.3.20+)
    # es. ".../releases/download/v0.3.23-cu124/llama_cpp_python-..."
    for durl_file in _glob.glob(
        str(lib_dir / "**" / "llama_cpp_python-*.dist-info" / "direct_url.json"), recursive=True
    ):
        try:
            data = _json.loads(Path(durl_file).read_text(encoding="utf-8"))
            url = data.get("url", "")
            if "-cu1" in url or "/cu1" in url:
                return True
        except Exception:
            pass

    # Pattern 2: libggml-cuda* in llama_cpp (layout 0.2.x / 0.3.x early)
    if _glob.glob(str(lib_dir / "**" / "llama_cpp" / "**" / "libggml-cuda*"), recursive=True):
        return True

    # Pattern 3: ldd su qualsiasi .so nel package llama_cpp (layout 0.3.x+)
    for so_file in _glob.glob(str(lib_dir / "**" / "llama_cpp" / "**" / "*.so"), recursive=True):
        try:
            r = subprocess.run(["ldd", so_file], capture_output=True, text=True, timeout=5)
            if any(kw in r.stdout for kw in ("libcudart", "libcublas")):
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return False


def _find_abetlen_wheel_url(cuda_tag: str) -> str | None:
    """Restituisce l'URL diretto del wheel CUDA più recente per linux_x86_64.

    L'indice abetlen esiste solo per cu121-cu124. I wheel da 0.3.20 in poi usano
    "py3-none-linux_x86_64.whl" senza +cuXYZ nel nome — l'unico modo per capire
    che è CUDA è l'URL stesso (es. .../v0.3.23-cu124/...).
    """
    import re as _re
    import urllib.request as _urlreq

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    url = f"https://abetlen.github.io/llama-cpp-python/whl/{cuda_tag}/llama-cpp-python/"

    try:
        with _urlreq.urlopen(url, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    candidates = []
    for m in _re.finditer(r'href="(https://[^"]*linux_x86_64\.whl)"', html):
        wheel_url = m.group(1)
        fname = wheel_url.rsplit("/", 1)[-1]
        # Accetta py3-none (universale, formato 0.3.20+) o wheel per la versione Python corrente
        if "-py3-none-" not in fname and f"-{py_tag}-" not in fname:
            continue
        ver_m = _re.search(r"llama_cpp_python-([0-9]+\.[0-9]+\.[0-9]+)", fname)
        if ver_m:
            ver_tuple = tuple(int(x) for x in ver_m.group(1).split("."))
            candidates.append((ver_tuple, wheel_url))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _install_llama_cpp_python_linux() -> bool:
    """Installa llama-cpp-python con CUDA su Linux/WSL. Ritorna True se GPU attivo.

    Strategie in ordine:
    1. Installa direttamente dall'URL del wheel CUDA pre-compilato su abetlen (cu121-cu124).
       L'URL diretto bypassa il problema di risoluzione versioni di uv/pip (che altrimenti
       preferirebbe il sorgente 0.3.23 da PyPI e builderebbe senza CUDA).
    2. Build da source con CMAKE_ARGS=-DGGML_CUDA=on  (richiede nvcc)
    3. Fallback CPU con warning esplicito
    """
    # Early exit: CUDA già presente
    if _llama_has_cuda_backend():
        print("  llama-cpp-python CUDA già installato ✓")
        return True

    cuda_tag = _detect_cuda_tag()  # es. "cu132"

    # Strategia 1 — wheel pre-compilato da URL diretto
    # abetlen pubblica wheel solo per cu121-cu124 (verificato).
    # Installa dall'URL diretto: uv/pip salva l'URL in direct_url.json nel dist-info,
    # permettendo rilevamento CUDA affidabile anche senza +cuXYZ nel nome del file.
    if cuda_tag:
        _KNOWN_TAGS = ["cu124", "cu123", "cu122", "cu121"]

        print(f"  CUDA {cuda_tag} rilevato — cerco wheel pre-compilati su abetlen (cu121–cu124)...")
        for candidate in _KNOWN_TAGS:
            wheel_url = _find_abetlen_wheel_url(candidate)
            if wheel_url is None:
                print(f"  ⚠️  Indice {candidate} non raggiungibile o senza wheel linux_x86_64 — skip")
                continue

            import re as _re
            ver_m = _re.search(r"llama_cpp_python-([0-9.]+)", wheel_url.rsplit("/", 1)[-1])
            ver_str = ver_m.group(1) if ver_m else "?"

            print(f"  Tentativo wheel {candidate} v{ver_str} (~1.3 GB — attendere)...")
            # Step 1: installa il wheel CUDA dall'URL diretto
            r = subprocess.run(
                ["uv", "pip", "install", "--reinstall", wheel_url],
                timeout=1800,  # 30 min — il wheel CUDA è ~1.3 GB
            )
            if r.returncode != 0:
                print(f"  ⚠️  Wheel {candidate} v{ver_str} fallito — provo il successivo...")
                continue
            # Step 2: installa le dipendenze del [server] extra da PyPI
            # senza reinstallare llama-cpp-python (già installato dal wheel CUDA)
            subprocess.run(
                ["uv", "pip", "install", f"llama-cpp-python[server]=={ver_str}"],
                timeout=120,
            )
            if _llama_has_cuda_backend():
                print(f"  llama-cpp-python CUDA ({candidate}, v{ver_str}) ✓")
                return True
            print(f"  ⚠️  Wheel {candidate} installato ma CUDA non verificato — provo il successivo...")
    else:
        print("  nvidia-smi non trovato — skip wheel, provo compilazione da source...")

    # Strategia 2 — build da source con nvcc
    has_nvcc = False
    try:
        has_nvcc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if has_nvcc:
        print("  nvcc disponibile — build da source con -DGGML_CUDA=on (alcuni minuti)...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
        env["FORCE_CMAKE"] = "1"
        r = subprocess.run(
            [str(_venv_python), "-m", "pip", "install",
             "llama-cpp-python[server]", "--no-cache-dir", "--force-reinstall", "--quiet"],
            env=env, timeout=600,
        )
        if r.returncode == 0 and _llama_has_cuda_backend():
            print("  llama-cpp-python CUDA (build sorgente) ✓")
            return True
        print("  ⚠️  Build da source fallita o senza backend CUDA.")

    # Strategia 3 — fallback CPU
    subprocess.run(["uv", "pip", "install", "llama-cpp-python[server]", "--reinstall"], check=True)
    print(
        "  ⚠️  llama-cpp-python installato in modalità CPU.\n"
        "      GPU attiva solo tramite il binario nativo (step successivo).\n"
        "      Per il fallback Python su GPU: installa il CUDA Toolkit in WSL e riesegui install.py."
    )
    return False


# ── 1. Verifica Python >= 3.10 ────────────────────────────────────────────────

step("Verifica versione Python")
if sys.version_info < (3, 10):
    _fail(f"Python >= 3.10 richiesto. Versione attuale: {sys.version}")
print(f"  Python {sys.version.split()[0]} ✓")


# ── 2. Verifica venv ─────────────────────────────────────────────────────────

step("Verifica ambiente virtuale")
print("  venv attivo ✓")


# ── 3. Installa dipendenze da requirements.txt ────────────────────────────────

step("Installazione dipendenze da requirements.txt")
req = ROOT / "requirements.txt"
if not req.exists():
    _fail("requirements.txt non trovato nella root del progetto")
subprocess.run(["uv", "pip", "install", "-r", str(req)], check=True)
print("  Dipendenze installate ✓")


# ── 4. Installa PyTorch con supporto CUDA ─────────────────────────────────────

step("Installazione PyTorch (CUDA)")

_torch_cuda_tag = _detect_cuda_tag()

if _torch_cuda_tag:
    # PyTorch supporta: cu121, cu124, cu126, cu128 — usa il più recente compatibile
    _TORCH_CUDA_TAGS = ["cu128", "cu126", "cu124", "cu121"]
    _detected_ver = int(_torch_cuda_tag[2:])
    _torch_tag = next(
        (t for t in _TORCH_CUDA_TAGS if int(t[2:]) <= _detected_ver),
        _TORCH_CUDA_TAGS[-1],
    )
    _torch_index = f"https://download.pytorch.org/whl/{_torch_tag}"
    print(f"  CUDA {_torch_cuda_tag} rilevato — installo PyTorch con indice {_torch_tag}")
    try:
        subprocess.run(
            ["uv", "pip", "install", "torch", "--index-url", _torch_index],
            check=True,
            timeout=300,
        )
        print(f"  PyTorch (CUDA {_torch_tag}) installato ✓")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  ⚠️  Installazione PyTorch CUDA fallita ({e}) — il sistema userà CPU per l'embedding")
else:
    print("  GPU NVIDIA non rilevata — PyTorch CPU già installato da requirements.txt ✓")


# ── 5. Installa llama-cpp-python ─────────────────────────────────────────────

step("Installazione llama-cpp-python")
if sys.platform == "win32":
    # Windows: il binario nativo usa DLL CUDA bundled — llama-cpp-python è solo fallback CPU.
    subprocess.run(["uv", "pip", "install", "llama-cpp-python[server]"], check=True)
    print("  llama-cpp-python (CPU fallback, Windows) ✓")
else:
    # Linux/WSL: tenta installazione GPU-first per garantire GPU anche nel fallback Python.
    _install_llama_cpp_python_linux()


def _download_llama_server_linux() -> bool:
    """Scarica llama-server precompilato Ubuntu+CUDA da llama.cpp GitHub releases.

    A differenza del binario Windows, è linkato dinamicamente alle librerie CUDA
    (libcudart, libcublas). In WSL2 si trovano in /usr/lib/wsl/lib/ e vengono
    aggiunte a LD_LIBRARY_PATH automaticamente da server_manager.py.

    Nelle release recenti (b5000+) il formato è .tar.gz (non più .zip) e il nome
    dell'asset non contiene necessariamente "cuda". Vengono preferiti in ordine:
    1. Asset con "cuda" esplicito nel nome (es. ubuntu-cuda-x64)
    2. Asset ubuntu-x64 generico (il .tar.gz principale, che può essere CUDA-linked)
    In entrambi i casi il binario viene verificato con ldd: se non è linkato contro
    CUDA viene rimosso, e Vesper usa il fallback llama-cpp-python CUDA wheel.
    """
    import json
    import tarfile
    import urllib.request
    import zipfile

    bin_dir = ROOT / "data" / "bin" / "llama-server"
    server_bin = bin_dir / "llama-server"

    if server_bin.exists():
        print("  llama-server (Linux) già presente ✓")
        return True

    print("  Recupero ultima release da GitHub (ggml-org/llama.cpp)...")
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
            headers={"Accept": "application/vnd.github+json", "User-Agent": "vesper-installer"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            release = json.loads(resp.read())
    except Exception as e:
        print(f"  ⚠️  GitHub API non raggiungibile: {e}\n"
              "  Avvia llama-server manualmente oppure imposta LLM_MANAGED=false nel .env.")
        return False

    tag = release.get("tag_name", "?")
    assets = release.get("assets", [])

    # Selezione asset Ubuntu x64: preferisce CUDA esplicito, poi il generico ubuntu-x64.
    # Supporta sia .zip (format pre-b5000) sia .tar.gz (format b5000+).
    # Esclude ARM, ROCm, SYCL, Vulkan, OpenVINO, openEuler (solo NVIDIA CUDA e CPU generico).
    _EXCLUDE = ("arm", "aarch", "rocm", "sycl", "vulkan", "openvino", "openeuler", "s390")

    def _is_candidate(name: str) -> bool:
        n = name.lower()
        if not ("ubuntu" in n and "x64" in n):
            return False
        if any(ex in n for ex in _EXCLUDE):
            return False
        return n.endswith(".zip") or n.endswith(".tar.gz")

    candidates_all = [a for a in assets if _is_candidate(a["name"])]
    # Preferisci asset con "cuda" nel nome, poi fallback al generico
    cuda_assets = [a for a in candidates_all if "cuda" in a["name"].lower()]
    chosen_asset = (
        sorted(cuda_assets, key=lambda x: x["name"], reverse=True)[0] if cuda_assets
        else (sorted(candidates_all, key=lambda x: x["name"], reverse=True)[0] if candidates_all else None)
    )

    if not chosen_asset:
        names = [a["name"] for a in assets]
        print(f"  ⚠️  Nessun asset Ubuntu x64 trovato nella release {tag}.\n"
              f"  Asset disponibili: {names}\n"
              "  Il server LLM userà llama-cpp-python (assicurati che sia installato con CUDA).")
        return False

    size_mb = chosen_asset["size"] // (1024 * 1024)
    asset_name = chosen_asset["name"]
    print(f"  Release {tag} — download {asset_name} ({size_mb} MB)...")
    bin_dir.mkdir(parents=True, exist_ok=True)
    archive_path = bin_dir / asset_name

    def _progress(count, block, total):
        if total > 0:
            pct = min(100, count * block * 100 // total)
            done = count * block // (1024 * 1024)
            print(f"\r  [{pct:3d}%] {done}/{total // (1024*1024)} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(chosen_asset["browser_download_url"], archive_path, _progress)
        print()
    except Exception as e:
        print(f"\n  ⚠️  Download fallito: {e}")
        return False

    print("  Estrazione...")
    import stat as _stat

    def _extract_member(fname: str, data: bytes) -> None:
        fname = Path(fname).name  # strip directory components
        if not fname:
            return
        if fname == "llama-server" or fname.endswith(".so") or ".so." in fname:
            target = bin_dir / fname
            target.write_bytes(data)
            if fname == "llama-server":
                target.chmod(target.stat().st_mode | _stat.S_IXUSR | _stat.S_IXGRP | _stat.S_IXOTH)

    if asset_name.endswith(".tar.gz") or asset_name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    fname = Path(member.name).name
                    f = tf.extractfile(member)
                    if f:
                        _extract_member(fname, f.read())
    else:
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                fname = Path(member.filename).name
                if fname and not member.is_dir():
                    _extract_member(fname, zf.read(member))

    archive_path.unlink(missing_ok=True)

    if not server_bin.exists():
        print("  ⚠️  llama-server non trovato dopo l'estrazione.")
        return False

    print(f"  llama-server estratto ({tag}) — verifica CUDA...")
    cuda_linked, cuda_found = _check_binary_cuda_linux(server_bin)

    if not cuda_linked:
        # Binario CPU-only: rimuovilo per non bloccare il fallback Python CUDA
        server_bin.unlink(missing_ok=True)
        print(
            "  ⚠️  Il binario Ubuntu x64 non è linkato contro CUDA (CPU-only).\n"
            "      llama.cpp non distribuisce più binari CUDA pre-compilati per Ubuntu.\n"
            "      Vesper userà llama-cpp-python con il wheel CUDA (step precedente)."
        )
        return False

    if cuda_linked and cuda_found:
        print(f"  llama-server pronto con CUDA ✓  ({tag})")
    else:
        # CUDA linkato ma librerie non trovate in questo ambiente (normale in WSL2)
        print(
            f"  llama-server estratto ✓  ({tag})\n"
            "  ⚠️  Librerie CUDA non trovate nella verifica post-installazione.\n"
            "      In WSL2 si trovano in /usr/lib/wsl/lib/ e vengono aggiunte\n"
            "      automaticamente a LD_LIBRARY_PATH all'avvio di Vesper."
        )

    return True


# ── 6. Download llama-server binario nativo ────────────────────────────────────

if not LLM_MANAGED:
    step("Download llama-server binario nativo")
    print("  LLM_MANAGED=false — skip download llama-server ✓")
    print("  Assicurati che il server LLM esterno sia raggiungibile all'avvio di Vesper.")
elif sys.platform == "win32":
    step("Download llama-server.exe (CUDA nativo, Windows)")
    _download_llama_server_windows()
else:
    step("Download llama-server (CUDA nativo, Linux/WSL)")
    _download_llama_server_linux()


# ── 7. Crea .env se non esiste ────────────────────────────────────────────────

step("Configurazione .env")
env_file = ROOT / ".env"
env_example = ROOT / ".env.example"
_env_just_created = False
if not env_file.exists():
    if not env_example.exists():
        _fail(".env.example non trovato — impossibile creare .env")
    shutil.copy(env_example, env_file)
    _env_just_created = True
    print(f"  .env creato da .env.example")
else:
    print("  .env già presente ✓")

if _env_just_created:
    print(f"\n  Apri '{env_file}' e configura almeno:")
    print("    TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, VAULT_PATH")
    print("\n  Puoi anche modificare LLM_MODEL_REPO / LLM_MODEL_FILE per scegliere")
    print("  un modello diverso da quello predefinito prima del download.")
    print("\n  Se vuoi gestire llama-server autonomamente (server esterno / inferenza remota),")
    print("  imposta LLM_MANAGED=false nel .env — i modelli LLM non verranno scaricati.")
    if sys.stdin.isatty():
        try:
            input("\n  Premi INVIO quando hai finito di configurare .env... ")
        except (EOFError, KeyboardInterrupt):
            print()
    else:
        print("\n  stdin non interattivo — continuo automaticamente.")
        print("  Configura .env manualmente e riesegui 'python install.py' se necessario.")
    # Ricarica le variabili dal .env appena configurato dall'utente
    load_dotenv(override=True)
    LLM_MODEL_REPO = os.getenv("LLM_MODEL_REPO", "unsloth/Qwen3.6-27B-GGUF")
    LLM_MODEL_FILE = os.getenv("LLM_MODEL_FILE", "Qwen3.6-27B-Q6_K.gguf")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    RERANKER_MODEL_REPO = os.getenv("RERANKER_MODEL_REPO", "Mungert/Qwen3-Reranker-4B-GGUF")
    RERANKER_MODEL_FILE = os.getenv("RERANKER_MODEL_FILE", "Qwen3-Reranker-4B-Q4_K_M.gguf")
    LLM_MANAGED = os.getenv("LLM_MANAGED", "true").lower() == "true"
    LLM_MODEL_DIR = ROOT / "data" / "models" / Path(LLM_MODEL_REPO)
    EMBEDDING_LOCAL_DIR = ROOT / "data" / "models" / Path(EMBEDDING_MODEL)
    RERANKER_MODEL_DIR = ROOT / "data" / "models" / Path(RERANKER_MODEL_REPO)
    VAULT_PATH = _abs_dir("VAULT_PATH", ROOT / "vault")
    DIRS_TO_CREATE[:] = [
        ROOT / "data" / "models",
        ROOT / "data" / "chroma",
        ROOT / "data" / "logs",
        VAULT_PATH / "raw",
        VAULT_PATH / "wiki",
        VAULT_PATH / "agenda" / "recurring",
        VAULT_PATH / "agenda" / "one-off",
    ]


# ── 8. Crea directory necessarie ──────────────────────────────────────────────

step("Creazione directory di progetto")
_vault_existed = VAULT_PATH.exists()
for d in DIRS_TO_CREATE:
    d.mkdir(parents=True, exist_ok=True)
if not _vault_existed:
    print(f"  ⚠️  Vault non trovato in '{VAULT_PATH}' — creato da zero.")
    print(f"      Se hai spostato il vault altrove, aggiorna VAULT_PATH in .env.")
print("  Directory create ✓")


# ── 9. Copia file di configurazione agente nel vault (solo se assenti) ────────

step("Copia file di configurazione agente nel vault")

_agent_files = ["AGENT.md", "USER.md"]
_src_prompts = ROOT / "src" / "prompts"
_copied_agent = _skipped_agent = 0

for _fname in _agent_files:
    _src = _src_prompts / _fname
    _dst = VAULT_PATH / _fname
    if not _src.exists():
        continue
    if _dst.exists():
        _skipped_agent += 1
    else:
        shutil.copy2(_src, _dst)
        _copied_agent += 1

_parts = []
if _copied_agent:
    _parts.append(f"{_copied_agent} file copiati")
if _skipped_agent:
    _parts.append(f"{_skipped_agent} già presenti (non sovrascritti)")
print(f"  {', '.join(_parts) or 'nessun file trovato in src/prompts'} ✓")


# ── 10. Download modello LLM ───────────────────────────────────────────────────

step(f"Download modello LLM: {LLM_MODEL_REPO} — {LLM_MODEL_FILE}")

_llm_model_target = LLM_MODEL_DIR / LLM_MODEL_FILE
if not LLM_MANAGED:
    print("  LLM_MANAGED=false — skip download modello LLM ✓")
    print("  Il modello è gestito dal server esterno configurato in LLM_BASE_URL.")
elif _llm_model_target.exists():
    print(f"  Modello già presente ✓")
else:
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        _fail("huggingface_hub non installato. Aggiungi huggingface_hub a requirements.txt e riprova.")

    hf_token: str | None = None
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip()
                if val:
                    hf_token = val
                break

    if hf_token:
        print("  Login HuggingFace con token...")
        login(token=hf_token)
    else:
        print("  HF_TOKEN non configurato. Download senza autenticazione (può essere più lento)")

    print(f"  Download in corso → {LLM_MODEL_DIR}")
    hf_hub_download(
        repo_id=LLM_MODEL_REPO,
        filename=LLM_MODEL_FILE,
        local_dir=str(LLM_MODEL_DIR),
        token=hf_token,
    )
    print("  Modello scaricato ✓")


# ── 11. Download modello di embedding ────────────────────────────────────────

step(f"Verifico modello embedding: {EMBEDDING_MODEL}")

if EMBEDDING_LOCAL_DIR.exists() and any(EMBEDDING_LOCAL_DIR.iterdir()):
    print(f"  Modello già presente ✓")
else:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _fail("sentence-transformers non installato. Aggiungi a requirements.txt e riprova.")
    print(f"  Download in corso → {EMBEDDING_LOCAL_DIR}")
    SentenceTransformer(EMBEDDING_MODEL).save(str(EMBEDDING_LOCAL_DIR))
    print("  Embedding scaricato ✓")

    _chroma_path = Path(os.getenv("CHROMA_PATH", str(ROOT / "data" / "chroma")))
    if _chroma_path.exists() and any(_chroma_path.iterdir()):
        print(
            f"\n  ⚠️  Indice ChromaDB esistente rilevato in '{_chroma_path}'.\n"
            "      Il nuovo modello produce embedding con dimensioni diverse dal precedente.\n"
            "      Dopo il riavvio esegui /reindex su Telegram per ricostruire l'indice."
        )


# ── 12. Download modello re-ranker ───────────────────────────────────────────

step(f"Download modello re-ranker: {RERANKER_MODEL_REPO} — {RERANKER_MODEL_FILE}")

if not LLM_MANAGED:
    print("  LLM_MANAGED=false — skip download re-ranker ✓")
    print("  Il re-ranker è gestito esternamente (avviato con llama-server su RERANKER_PORT se necessario).")
elif not RERANKER_ENABLED:
    print("  RERANKER_ENABLED=false — skip ✓")
elif (RERANKER_MODEL_DIR / RERANKER_MODEL_FILE).exists():
    print("  Modello già presente ✓")
else:
    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError:
        _fail("huggingface_hub non installato. Aggiungi huggingface_hub a requirements.txt e riprova.")

    hf_token_reranker: str | None = None
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                val = line.split("=", 1)[1].strip()
                if val:
                    hf_token_reranker = val
                break

    if hf_token_reranker:
        login(token=hf_token_reranker)

    RERANKER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Download in corso → {RERANKER_MODEL_DIR}")
    try:
        hf_hub_download(
            repo_id=RERANKER_MODEL_REPO,
            filename=RERANKER_MODEL_FILE,
            local_dir=str(RERANKER_MODEL_DIR),
            token=hf_token_reranker,
        )
    except Exception as e:
        print(
            f"\n  ⚠️  Repo del modello re-ranker non raggiungibile:\n\t'{RERANKER_MODEL_REPO / RERANKER_MODEL_FILE}'.\n"
            "      Assicurati di aver specificato il repo corretto nel file .env\n"
            "      ed esegui nuovamente 'python install.py' per scaricarlo."
        )
        sys.exit(1)
    print("  Modello re-ranker scaricato ✓")


# ── 13. Riepilogo finale ─────────────────────────────────────────────────────

if LLM_MANAGED:
    print("""
✓ Installazione completata

▶ Prossimi passi:
   1. Compila .env se non l'hai ancora fatto (TELEGRAM_BOT_TOKEN, TELEGRAM_ADMIN_CHAT_ID, VAULT_PATH)
   2. Avvia l'agente con 'python main.py'
""")
else:
    print("""
✓ Installazione completata  (LLM_MANAGED=false)

▶ Prossimi passi:
   1. Avvia il tuo server LLM esterno e verifica che LLM_BASE_URL nel .env punti all'endpoint corretto
   2. Avvia l'agente con 'python main.py'

   Per passare alla gestione automatica del server LLM, imposta LLM_MANAGED=true
   nel .env e riesegui 'python install.py' per scaricare i modelli.
""")
