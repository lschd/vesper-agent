# Vesper su Docker (GPU / CUDA)

Esecuzione containerizzata con llama.cpp accelerato via CUDA. Il CUDA toolkit
viene usato **solo durante la build** dell'immagine (per compilare
`llama-cpp-python` da source): non va installato sull'host e non va ricompilato
ad ogni avvio. `llama-cpp-python` è buildato con baseline CPU **AVX2 senza
AVX512**, quindi gira anche su CPU consumer (es. Intel i9-14900) senza SIGILL.

## Prerequisiti host

1. **Driver NVIDIA** recente (CUDA ≥ 12.x). Su WSL2 si installa lato Windows.
   Verifica: `nvidia-smi`.
2. **Docker** + **Docker Compose** v2.
3. **NVIDIA Container Toolkit** (per il passthrough GPU nei container):
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker   # oppure: sudo service docker restart
   ```
   Verifica passthrough:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu24.04 nvidia-smi
   ```

## Setup

1. **`.env`** — copia `.env.example` in `.env` e compila almeno `TELEGRAM_BOT_TOKEN`,
   `TELEGRAM_ADMIN_CHAT_ID`, e `HF_TOKEN` se il modello lo richiede. Mantieni
   `LLM_MANAGED=true` (i server LLM e re-ranker girano dentro il container).

2. **Dati e modelli** — i dati interni (modelli, indice ChromaDB, log,
   `conversations.db`) vivono in un **volume Docker permanente** chiamato
   `vesper-data`, montato su `/app/data`. Il volume sopravvive a
   `docker compose down` e alla rimozione del container.

   Popolalo una volta coi modelli (scarica sull'host, poi copia nel volume):
   ```bash
   python install.py                       # scarica LLM, embedding e re-ranker in ./data/models/
   docker volume create vesper-data
   docker run --rm -v vesper-data:/dest -v "$(pwd)/data":/src:ro \
     alpine cp -a /src/. /dest/            # copia ./data nel volume permanente
   ```
   Per ispezionare il volume in seguito:
   ```bash
   docker run --rm -v vesper-data:/d alpine ls -R /d/models
   ```

3. **Vault** — la cartella `./vault/` è un **bind mount** sull'host, quindi puoi
   aggiungere documenti direttamente: metti PDF/DOCX in `./vault/raw/` o note
   Markdown in `./vault/wiki/`. Il watchdog li rileva e li (re)indicizza in automatico.

## Build & avvio

```bash
docker compose build      # compila l'immagine (la build di llama-cpp-python CUDA richiede qualche minuto)
docker compose up -d      # avvia in background
docker compose logs -f    # segui i log
```

- Web UI / API FastAPI: http://localhost:8080
- I server LLM (`8000`) e re-ranker (`8001`) restano interni al container.

Stop / riavvio:
```bash
docker compose down
docker compose up -d
```

## GPU diversa dalla A2000

L'immagine è compilata per la compute capability **8.6** (RTX A2000 / Ampere).
Per un'altra GPU passa l'arch corretta in build (es. `89` per Ada, `90` per Hopper):

```bash
docker compose build --build-arg CUDA_ARCH=89
```
oppure modifica `args.CUDA_ARCH` in `docker-compose.yml`.

## Note

- **Modello LLM grande**: `Qwen3.6-27B-Q6_K` (~22 GB) non entra interamente nei
  12 GB della A2000 → offload parziale su CPU (gestito automaticamente). Per stare
  tutto in VRAM usa un quant più piccolo (es. `Q4_K_M`).
- I dati persistono nel volume Docker `vesper-data` (modelli, indice ChromaDB,
  log) e nel bind mount `./vault` (note). L'immagine non li contiene.
- L'embedding (`sentence-transformers`/PyTorch) usa la GPU in modo indipendente da
  llama.cpp.
