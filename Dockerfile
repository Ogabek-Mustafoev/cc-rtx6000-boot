# cc-rtx6000 boot-image — venv'lar va cuda-13 toolchain IMAGE ICHIDA.
#
# Nega: RunPod'ning RTX Pro 6000 hostlarida /workspace tarmoq-FUSE (mfs) va vLLM'ning
# ~37k faylini har protsess-startda tarmoqdan o'qish 8-13 daqiqa yeydi (2026-08-23
# o'lchovi; model-og'irliklar esa ~1 soniya). Image esa hostning LOKAL diskiga
# tortiladi va layer-keshlaanadi — importlar NVMe tezlikda, boot ~3-5 daqiqaga tushadi.
#
# Image'da NIMA YO'Q (ataylab): modellar (23GB judge + 1.2GB reranker) — ular
# /workspace/cc-rtx/hf keshida qoladi (prefetch bilan ~4 daq, bir marta yuklangan);
# kod (server/launch/tools — rsync bilan yangilanadi, image-rebuild'siz); env.sh (sirlar).
#
# Build (x86_64; Mac'da buildx+qemu bilan ham bo'ladi, sekinroq):
#   docker buildx build --platform linux/amd64 -t <registry>/cc-rtx6000-boot:v1 \
#     -f launch/Dockerfile launch/ --push
# Pod'da: imageName=<registry>/cc-rtx6000-boot:v1; launcher'lar CC_VENV_ROOT=/opt/cc
# ni avtomatik afzal ko'radi (mavjud bo'lsa).

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# cuda-13 toolchain — konteyner-resetda yo'qolmasligi uchun image-qatlamda
# (start.sh'ning "toolchain missing — installing" 5-daqiqasi yo'qoladi).
# To'liq cuda-toolkit-13-0 EMAS (~10GB: gdb/sanitizer/docs — GH-runner diskini
# to'ldirib yubordi, 2026-08-24 birinchi build shu bilan yiqildi): flashinfer-JIT'ga
# yetadigan to'plam — nvcc + cudart-headers + cccl (cub/thrust) + profiler-api (~3GB).
# Yetishmagan komponent chiqsa vllm.log'da baland ko'rinadi va bu ro'yxatga qo'shiladi.
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-cccl-13-0 cuda-profiler-api-13-0 \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

# UV_NO_CACHE: wheel-kesh venv bilan birga har baytni IKKI marta saqlaydi (~13GB) —
# 2-build aynan shu tufayli "no space"da yiqildi; image-qatlamda keshning o'rni yo'q.
ENV UV_LINK_MODE=copy UV_NO_CACHE=1
RUN pip install --break-system-packages --no-cache-dir uv

# venv-app: torch-siz, tez.
RUN python3 -m venv /opt/cc/venv-app \
    && /opt/cc/venv-app/bin/pip install --no-cache-dir \
       fastapi "uvicorn[standard]" httpx pillow anyio openai pytest

# venv-vllm: setup.sh bilan AYNAN bir xil juftlik (SM120-tasdiqlangan; flashinfer
# vllm'dan KEYIN — 0.6.14-pinini 0.6.17 bilan yopish uchun, fp8_gemm segfault davosi).
RUN uv venv /opt/cc/venv-vllm --python 3.12 \
    && uv pip install --python /opt/cc/venv-vllm/bin/python vllm==0.26.0 ninja \
    && uv pip install --python /opt/cc/venv-vllm/bin/python flashinfer-python==0.6.17

# Sog'lomlik-belgisi: launcher'lar shu marker orqali image-venv borligini biladi.
RUN /opt/cc/venv-vllm/bin/python -c "import vllm" \
    && /opt/cc/venv-app/bin/python -c "import fastapi, httpx" \
    && touch /opt/cc/.venvs-baked
