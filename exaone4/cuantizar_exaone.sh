#!/usr/bin/env bash
# =============================================================================
#  Lumi · Fase 2 · Cuantización de EXAONE 4.0 1.2B (llama.cpp / GGUF)
#  Correr en la Terminal de macOS (Apple Silicon, usa Metal), desde esta carpeta:
#
#     cd ~/Desktop/ProyectoGrado/proyecto/lumi_project/model_quantization_phase2/exaone4
#     bash cuantizar_exaone.sh           # pipeline completo (~40-70 min)
#     bash cuantizar_exaone.sh eval      # solo re-evaluar
#     bash cuantizar_exaone.sh limpiar   # borra intermedios pesados (deja los GGUF finales)
#
#  Calibración y evaluación salen de lumi_project/calibracion_corregido.json
#  (300 conversaciones del tutor con respuestas corregidas a mano), renderizadas con el
#  system prompt real de NubeEngine y la plantilla de chat de EXAONE.
#  Partición estratificada por categoría (semilla 42): 80 % calibración (imatrix) · 20 % evaluación.
#  Los ids son los mismos que en calibracion.json, así que la partición es idéntica a la anterior.
#
#  Variantes:  F16 (referencia) · Q8_0 · Q6_K · Q5_K_M-imat · Q4_K_M · Q4_K_M-imat · Q4_0 · Q4_0-imat
#
#  Si cambias el archivo de calibración, el script lo detecta (hash SHA-256) y rehace
#  los datos, la imatrix, las variantes -imat y la evaluación.
# =============================================================================
set -euo pipefail

MODEL_ID="LGAI-EXAONE/EXAONE-4.0-1.2B"
NAME="exaone4-1.2b"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"                  # lumi_project/
CALIB_JSON="${CALIB_JSON:-$PROJ/calibracion_corregido.json}"
DART="$PROJ/flutter_app/lib/data/services/tutor/nube_engine.dart"
RENDER_V="v3"                     # versión del renderizado (ver prepare_data)

WORK="$HERE/work"                 # todo lo pesado vive aquí (está en .gitignore)
LLAMA="$WORK/llama.cpp"
BIN="$LLAMA/build/bin"
HF_DIR="$WORK/hf/$NAME"
OUT="$WORK/gguf"
DATA="$HERE/data"                 # textos renderizados + ids de la partición (sí se versiona)
LOGS="$HERE/logs"
VENV="$WORK/.venv"
THREADS="${THREADS:-4}"
IMATRIX_CTX="${IMATRIX_CTX:-4096}"   # cabe el system prompt (~1.3k tokens) + la conversación completa
EVAL_CHUNKS="${EVAL_CHUNKS:-}"    # vacío = todo data/eval.txt
# se pueden elegir:  VARIANTES="Q8_0 Q4_K_M Q4_K_M-imat Q4_0-imat" bash ...
read -r -a VARIANTES <<< "${VARIANTES:-Q8_0 Q6_K Q5_K_M-imat Q4_K_M Q4_K_M-imat Q4_0 Q4_0-imat}"

mkdir -p "$WORK" "$OUT" "$DATA" "$LOGS"
log(){ printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }

# -----------------------------------------------------------------------------
# 0. Requisitos y espacio en disco
# -----------------------------------------------------------------------------
check_disk(){
  local libre_gb
  libre_gb=$(df -g "$HERE" | awk 'NR==2{print $4}')
  echo "Espacio libre: ${libre_gb} GB (se necesitan ~13 GB; al final quedan ~7 GB ocupados)"
  if [ "$libre_gb" -lt 13 ]; then
    echo "⚠  Poco espacio. Libera disco o corre con EVAL_CHUNKS=15 para reducir el archivo .kld."
    read -r -p "¿Continuar de todas formas? [s/N] " r; [[ "$r" =~ ^[sS]$ ]] || exit 1
  fi
}

setup(){
  log "Verificando requisitos"
  [ -f "$CALIB_JSON" ] || { echo "No encuentro $CALIB_JSON"; exit 1; }
  [ -f "$DART" ]       || { echo "No encuentro $DART"; exit 1; }
  xcode-select -p >/dev/null 2>&1 || { echo "Instala las Command Line Tools: xcode-select --install"; exit 1; }
  command -v cmake  >/dev/null || { echo "Falta cmake: brew install cmake"; exit 1; }
  command -v python3 >/dev/null || { echo "Falta python3"; exit 1; }
  check_disk

  if [ ! -d "$LLAMA" ]; then
    log "Clonando llama.cpp"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
  fi
  ( cd "$LLAMA" && git log -1 --format='llama.cpp commit: %h (%cd)' --date=short ) | tee "$LOGS/llama_cpp_version.txt"

  if [ ! -x "$BIN/llama-server" ]; then
    log "Compilando llama.cpp (Metal activado por defecto en macOS)"
    cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build "$LLAMA/build" -j --config Release \
      --target llama-quantize llama-imatrix llama-perplexity llama-bench llama-cli llama-server
  fi

  [ -d "$VENV" ] || { log "Creando entorno de Python"; python3 -m venv "$VENV"; }
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -q --upgrade pip
  pip install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"
  # transformers 4.x exige huggingface_hub <1.0; el CLI `hf` viene desde 0.34
  pip install -q "huggingface_hub[cli]>=0.34.0,<1.0"
}

# -----------------------------------------------------------------------------
# Invalida lo que depende de la calibración si el archivo cambió
# -----------------------------------------------------------------------------
check_calib(){
  local nuevo viejo="" rv=""
  nuevo=$(shasum -a 256 "$CALIB_JSON" | awk '{print $1}')
  if [ -f "$DATA/fuente.txt" ]; then
    viejo=$(awk -F': ' '/^sha256/{print $2}' "$DATA/fuente.txt")
    rv=$(awk -F': ' '/^render/{print $2}' "$DATA/fuente.txt")
  fi
  if [ -f "$DATA/calib.txt" ] && { [ "$viejo" != "$nuevo" ] || [ "$rv" != "$RENDER_V" ]; }; then
    log "Cambió la calibración o el renderizado: se rehacen datos, imatrix, variantes -imat y evaluación"
    rm -f "$DATA/calib.txt" "$DATA/eval.txt" "$DATA/particion.json" \
          "$OUT/$NAME.imatrix" "$OUT/$NAME-"*-imat.gguf "$OUT/$NAME-F16.kld" \
          "$LOGS"/kld_*.log "$LOGS"/ppl_F16.log "$LOGS"/imatrix.log "$LOGS"/bench_cpu_*-imat.csv
  fi
}

# -----------------------------------------------------------------------------
# 1. Descargar y convertir a GGUF F16
# -----------------------------------------------------------------------------
convert(){
  source "$VENV/bin/activate"
  if [ ! -f "$OUT/$NAME-F16.gguf" ]; then
    if [ ! -f "$HF_DIR/config.json" ]; then
      log "Descargando $MODEL_ID (~2.6 GB)"
      hf download "$MODEL_ID" --local-dir "$HF_DIR"
    fi
    log "Convirtiendo a GGUF F16"
    python "$LLAMA/convert_hf_to_gguf.py" "$HF_DIR" --outtype f16 --outfile "$OUT/$NAME-F16.gguf.tmp"
    mv "$OUT/$NAME-F16.gguf.tmp" "$OUT/$NAME-F16.gguf"
    rm -f "$HF_DIR"/*.safetensors && echo "Safetensors borrados (ya está el F16)"
  fi
}

# -----------------------------------------------------------------------------
# 2. Renderizar conversaciones de Lumi (system prompt real + plantilla de EXAONE)
# -----------------------------------------------------------------------------
prepare_data(){
  source "$VENV/bin/activate"
  if [ -f "$DATA/calib.txt" ] && [ -f "$DATA/eval.txt" ]; then log "Datos ya preparados"; return; fi
  [ -f "$HF_DIR/tokenizer_config.json" ] || hf download "$MODEL_ID" --local-dir "$HF_DIR" \
      --include "tokenizer*" "*.json"
  log "Renderizando $(basename "$CALIB_JSON") → data/calib.txt + data/eval.txt"
  python - "$CALIB_JSON" "$DART" "$HF_DIR" "$DATA" "$RENDER_V" <<'PY'
import json, re, random, sys, pathlib, hashlib
from collections import defaultdict
from transformers import AutoTokenizer

calib, dart, hf, dst, render_v = sys.argv[1], sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4]), sys.argv[5]
prompt = re.search(r"_systemPrompt = '''(.*?)''';", open(dart, encoding="utf-8").read(), re.S).group(1)
todas = json.load(open(calib, encoding="utf-8"))
convs = [c for c in todas if c.get("messages") and not c.get("invalidos")]
print(f"{len(todas)} conversaciones en el archivo · {len(convs)} válidas")

malas = []
for c in convs:
    for m in c["messages"]:
        if m["role"] != "assistant":
            continue
        try:
            d = json.loads(m["content"])
            ok = isinstance(d, dict) and isinstance(d.get("guia"), str) and d["guia"].strip()
        except json.JSONDecodeError:
            ok = False
        if not ok:
            malas.append(c["id"]); break
if malas:
    print(f"⚠  {len(malas)} conversaciones tienen respuestas que no son JSON válido de Lumi: {malas[:10]}")

# Partición estratificada por categoría (semilla fija → reproducible)
random.seed(42)
by = defaultdict(list)
for c in convs:
    by[c.get("categoria")].append(c)
train, test = [], []
for cat in sorted(by, key=str):
    lst = sorted(by[cat], key=lambda c: c["id"])
    random.shuffle(lst)
    k = max(1, round(len(lst) * 0.2))
    test += lst[:k]; train += lst[k:]

tok = AutoTokenizer.from_pretrained(hf)
def render(c, con_sistema):
    msgs = ([{"role": "system", "content": prompt}] if con_sistema else []) + \
           [{"role": m["role"], "content": m["content"]} for m in c["messages"] if m["role"] in ("user", "assistant")]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False)

assert prompt[:80] in render(train[0], True), "El system prompt no quedó en el texto renderizado: revisa la plantilla"

# Renderizado v3.
#   calib: TODAS las conversaciones llevan el system prompt, como en la app, y la imatrix se
#          calcula con contexto de 4096 tokens: cada respuesta del tutor se procesa con su prompt
#          delante, que es justo la capacidad que hay que proteger al cuantizar.
#          (v1 usaba contexto de 512: el prompt, de ~1.3k tokens, nunca quedaba junto a la respuesta.)
#   eval:  sin system prompt → PPL y KLD miden el diálogo; si se incluyera, la mayoría de los
#          tokens puntuados serían copias del prompt, fáciles de predecir, y la métrica se inflaría.
# Que el modelo SIGA el prompt se mide en la evaluación pedagógica (prompt completo + juez).
textos = {"calib": "\n\n".join(render(c, True) for c in train),
          "eval":  "\n\n".join(render(c, False) for c in test)}
for nombre, txt in textos.items():
    (dst / f"{nombre}.txt").write_text(txt, encoding="utf-8")
    ntok = len(tok(txt)["input_ids"])
    share = 100 * txt.count(prompt) * len(prompt) / len(txt)
    print(f"{nombre}: {len(train if nombre == 'calib' else test)} conversaciones · {ntok:,} tokens · "
          f"system prompt = {share:.0f} % del texto")
(dst / "particion.json").write_text(json.dumps(
    {"semilla": 42, "calib": sorted(c["id"] for c in train), "eval": sorted(c["id"] for c in test)},
    ensure_ascii=False, indent=1), encoding="utf-8")
sha = hashlib.sha256(open(calib, "rb").read()).hexdigest()
(dst / "fuente.txt").write_text(f"archivo: {pathlib.Path(calib).name}\nsha256: {sha}\nrender: {render_v}\n",
                                encoding="utf-8")

otra = dst.parent.parent / "qwen3.2" / "data" / "particion.json"
if otra.exists():
    igual = json.load(open(otra))["eval"] == sorted(c["id"] for c in test)
    print("Partición igual a la de Qwen3.5:", "sí" if igual else "NO ⚠")
PY
}

# -----------------------------------------------------------------------------
# 3. imatrix con las conversaciones de calibración
# -----------------------------------------------------------------------------
imatrix(){
  [ -f "$OUT/$NAME.imatrix" ] && { log "imatrix ya existe"; return; }
  log "Calculando imatrix con las conversaciones corregidas de Lumi"
  "$BIN/llama-imatrix" -m "$OUT/$NAME-F16.gguf" -f "$DATA/calib.txt" \
    -o "$OUT/$NAME.imatrix.tmp" -c "$IMATRIX_CTX" -b "$IMATRIX_CTX" -ngl 99 -t "$THREADS" 2>&1 | tee "$LOGS/imatrix.log"
  mv "$OUT/$NAME.imatrix.tmp" "$OUT/$NAME.imatrix"
}

# -----------------------------------------------------------------------------
# 4. Cuantizar
# -----------------------------------------------------------------------------
q(){ # $1 tipo · $2 sufijo · $3 usar imatrix (0/1)
  local f="$OUT/$NAME-$2.gguf"
  [ -f "$f" ] && return
  log "Cuantizando $2"
  # se escribe a .tmp y solo se renombra si termina: una interrupción no deja un GGUF truncado
  if [ "$3" = 1 ]; then
    "$BIN/llama-quantize" --imatrix "$OUT/$NAME.imatrix" "$OUT/$NAME-F16.gguf" "$f.tmp" "$1" "$THREADS"
  else
    "$BIN/llama-quantize" "$OUT/$NAME-F16.gguf" "$f.tmp" "$1" "$THREADS"
  fi > "$LOGS/quantize_$2.log" 2>&1
  mv "$f.tmp" "$f"
  tail -2 "$LOGS/quantize_$2.log"
}
quantize(){
  for v in "${VARIANTES[@]}"; do
    if [ "${v%-imat}" != "$v" ]; then q "${v%-imat}" "$v" 1; else q "$v" "$v" 0; fi
  done
}

# -----------------------------------------------------------------------------
# 5. Evaluar en conversaciones NO vistas: PPL, KL vs F16, velocidad CPU, tamaño
# -----------------------------------------------------------------------------
evaluate(){
  local chunks=()
  [ -n "$EVAL_CHUNKS" ] && chunks=(--chunks "$EVAL_CHUNKS")
  if [ ! -f "$OUT/$NAME-F16.kld" ]; then
    log "Logits de referencia F16 sobre data/eval.txt"
    "$BIN/llama-perplexity" -m "$OUT/$NAME-F16.gguf" -f "$DATA/eval.txt" -c 512 \
      ${chunks[@]+"${chunks[@]}"} -ngl 99 -t "$THREADS" \
      --kl-divergence-base "$OUT/$NAME-F16.kld.tmp" 2>&1 | tee "$LOGS/ppl_F16.log"
    mv "$OUT/$NAME-F16.kld.tmp" "$OUT/$NAME-F16.kld"
  fi
  for v in "${VARIANTES[@]}"; do
    if grep -q "Mean    KLD" "$LOGS/kld_$v.log" 2>/dev/null && [ -s "$LOGS/bench_cpu_$v.csv" ]; then
      echo "$v ya evaluada (se omite)"; continue
    fi
    log "Evaluando $v"
    "$BIN/llama-perplexity" -m "$OUT/$NAME-$v.gguf" -c 512 -ngl 99 -t "$THREADS" \
      --kl-divergence-base "$OUT/$NAME-F16.kld" --kl-divergence 2>&1 | tee "$LOGS/kld_$v.log" | tail -12 \
      || echo "⚠  llama-perplexity falló con $v (revisa logs/kld_$v.log); se sigue con las demás"
    "$BIN/llama-bench" -m "$OUT/$NAME-$v.gguf" -p 512 -n 128 -t "$THREADS" -ngl 0 -o csv \
      > "$LOGS/bench_cpu_$v.csv" 2>/dev/null || true
  done

  source "$VENV/bin/activate"
  python - "$OUT" "$LOGS" "$NAME" "$HERE" "${VARIANTES[@]}" <<'PY'
import sys, re, csv, pathlib
out, logs, name, here = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], pathlib.Path(sys.argv[4])
variantes = sys.argv[5:]
def grab(pat, txt):
    m = re.search(pat, txt); return float(m.group(1)) if m else None
ref = (logs/"ppl_F16.log").read_text(errors="ignore") if (logs/"ppl_F16.log").exists() else ""
rows = [("F16", (out/f"{name}-F16.gguf").stat().st_size/1e9, grab(r"Final estimate: PPL = ([\d.]+)", ref), 0.0, 100.0, None, None)]
for v in variantes:
    p = logs/f"kld_{v}.log"
    t = p.read_text(errors="ignore") if p.exists() else ""
    ppl  = grab(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", t) or grab(r"Final estimate: PPL = ([\d.]+)", t)
    kld  = grab(r"Mean\s+KLD:\s*([\d.]+)", t)
    same = grab(r"Same top p:\s*([\d.]+)", t)
    pp = tg = None
    b = logs/f"bench_cpu_{v}.csv"
    if b.exists():
        for r in csv.DictReader(b.open()):
            ts = float(r.get("avg_ts", 0) or 0)
            if r.get("n_prompt") not in (None, "0"): pp = ts
            elif r.get("n_gen") not in (None, "0"): tg = ts
    g = out/f"{name}-{v}.gguf"
    rows.append((v, g.stat().st_size/1e9 if g.exists() else None, ppl, kld, same, pp, tg))
hdr = ["variante","tamaño_GB","PPL_lumi","KLD_media_vs_F16","top1_igual_%","prompt_tok_s_CPU","gen_tok_s_CPU"]
dest = here/"resultados_exaone.csv"
with open(dest, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
fmt = lambda x: "-" if x is None else (f"{x:.4f}" if isinstance(x, float) and x < 1 else f"{x:.2f}" if isinstance(x, float) else x)
print("\n" + " | ".join(hdr))
for r in rows: print(" | ".join(fmt(c) for c in r))
print(f"\nGuardado en {dest}")
PY
}

limpiar(){
  log "Borrando intermedios pesados (safetensors, .kld). Se conservan los GGUF."
  rm -rf "$HF_DIR" "$OUT/$NAME-F16.kld"
  read -r -p "¿Borrar también el F16 (2.6 GB)? Lo necesita la evaluación pedagógica de F16. [s/N] " r
  [[ "$r" =~ ^[sS]$ ]] && rm -f "$OUT/$NAME-F16.gguf"
  du -sh "$WORK"
}

case "${1:-all}" in
  all)     setup; check_calib; convert; prepare_data; imatrix; quantize; evaluate
           rm -rf "$HF_DIR"; log "Listo. Safetensors borrados para liberar espacio."
           echo "Siguiente paso: bash evaluar_pedagogico.sh" ;;
  eval)    evaluate ;;
  limpiar) limpiar ;;
  *)       echo "Uso: bash $0 [all|eval|limpiar]"; exit 1 ;;
esac
