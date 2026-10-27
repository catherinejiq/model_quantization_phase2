#!/usr/bin/env bash
# =============================================================================
#  Lumi · Fase 2 · Evaluación pedagógica de los GGUF de EXAONE con el juez
#
#  Para cada variante: levanta llama-server con el GGUF, reproduce los 60 guiones
#  de la partición de evaluación (data/particion.json) con tools/correr_guiones.py
#  (mismo prompt y parámetros que NubeEngine) y los califica con tools/juez.py
#  (GPT-5 por OpenRouter). Compara contra la nube (evaluacion_calibracion.jsonl).
#
#  Uso (desde model_quantization_phase2/):
#     bash evaluar_pedagogico.sh                              # F16 Q4_K_M-imat Q4_0-imat
#     bash evaluar_pedagogico.sh F16 Q5_K_M-imat Q4_0-imat    # las que quieras
#     bash evaluar_pedagogico.sh --resumen                    # solo la tabla final
#
#  Es reanudable: los guiones y calificaciones ya hechos se omiten.
#  Costo del juez: ~60 llamadas a GPT-5 por variante.
# =============================================================================
set -euo pipefail

NAME="exaone4-1.2b"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
BIN="$HERE/work/llama.cpp/build/bin"
OUT="$HERE/work/gguf"
RES="$HERE/resultados_pedagogicos"
LOGS="$HERE/logs"
PORT="${PORT:-8091}"
SLOTS="${SLOTS:-4}"                 # conversaciones en paralelo
JUEZ_WORKERS="${JUEZ_WORKERS:-6}"
PY="${PY:-python3}"
mkdir -p "$RES" "$LOGS"
log(){ printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }

SERVER_PID=""
parar_servidor(){ if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; SERVER_PID=""; }
trap parar_servidor EXIT INT TERM

# IDs de evaluación (las 60 conversaciones que no se usaron para calibrar)
EVAL_IDS=$("$PY" -c "import json;print(' '.join(json.load(open('$HERE/data/particion.json'))['eval']))")
N_EVAL=$(wc -w <<<"$EVAL_IDS" | tr -d ' ')

levantar_servidor(){ # $1 = variante
  local modelo="$OUT/$NAME-$1.gguf"
  [ -f "$modelo" ] || { echo "No existe $modelo (corre primero cuantizar_exaone.sh)"; exit 1; }
  log "Levantando llama-server con $1"
  # La plantilla embebida en el GGUF usa {%- continue %} y el motor Jinja de llama.cpp descarta
  # el mensaje de sistema; esta copia es idéntica pero sin continue (ver apply-template).
  "$BIN/llama-server" -m "$modelo" --port "$PORT" -ngl 99 \
    -c $((8192 * SLOTS)) -np "$SLOTS" --jinja \
    --chat-template-file "$HERE/exaone4_chat_template.jinja" \
    --chat-template-kwargs '{"enable_thinking": false}' \
    > "$LOGS/server_$1.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "Servidor listo"; return; }
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "llama-server falló; revisa $LOGS/server_$1.log"; tail -20 "$LOGS/server_$1.log"; exit 1; }
    sleep 1
  done
  echo "llama-server no respondió en 120 s"; exit 1
}

correr_variante(){ # $1 = variante
  local v="$1" guiones="$RES/guiones_$1.json" juez="$RES/juez_$1.jsonl"
  local hechos=0
  [ -f "$guiones" ] && hechos=$("$PY" -c "import json;print(len(json.load(open('$guiones'))))" 2>/dev/null || echo 0)

  if [ "$hechos" -lt "$N_EVAL" ]; then
    levantar_servidor "$v"
    log "Reproduciendo $N_EVAL guiones con $v ($hechos ya hechos)"
    local ids=(); for i in $EVAL_IDS; do ids+=(--id "$i"); done
    ( cd "$PROJ" && LUMI_TUTOR_URL="http://127.0.0.1:$PORT/v1/chat/completions" LUMI_TUTOR_TIMEOUT=300 \
        "$PY" tools/correr_guiones.py guiones.json "${ids[@]}" \
        --modelo "$NAME-$v" --workers "$SLOTS" -o "$guiones" ) || true
    parar_servidor
  else
    echo "$v: guiones completos ($hechos/$N_EVAL)"
  fi

  local calificados=0
  [ -f "$juez" ] && calificados=$(grep -c '"score"' "$juez" || true)
  if [ "$calificados" -lt "$hechos" ] || [ "$calificados" -eq 0 ]; then
    log "Calificando $v con el juez (GPT-5)"
    ( cd "$PROJ" && "$PY" tools/juez.py "$guiones" --workers "$JUEZ_WORKERS" -o "$juez" ) | tail -20
  else
    echo "$v: ya calificada ($calificados)"
  fi
}

resumen(){
  log "Resumen"
  "$PY" - "$HERE" "$PROJ" "$RES" <<'PY'
import json, sys, csv, pathlib
from collections import defaultdict
here, proj, res = map(pathlib.Path, sys.argv[1:4])
ev = set(json.load(open(here / "data/particion.json"))["eval"])

def leer_jsonl(p):
    filas = []
    for l in p.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(l)
            if isinstance(d, dict) and "score" in d:
                filas.append(d)
        except json.JSONDecodeError:
            pass  # líneas de resumen pegadas al final, etc.
    return filas

fuentes = {"nube (Qwen3-VL-235B)": (leer_jsonl(proj / "evaluacion_calibracion.jsonl"), None)}
for j in sorted(res.glob("juez_*.jsonl")):
    v = j.stem[len("juez_"):]
    g = res / f"guiones_{v}.json"
    fuentes[v] = (leer_jsonl(j), json.load(open(g)) if g.exists() else None)

criterios = ["no_revela_respuesta", "no_calcula_por_estudiante", "valida_correctamente",
             "aritmetica_correcta", "cierra_al_acertar", "evita_fallo_objetivo"]
hdr = ["variante", "n", "score", "formato_invalido_%"] + criterios
filas_csv = []
for nombre, (filas, convs) in fuentes.items():
    filas = [f for f in filas if f["id"] in ev]
    if not filas:
        continue
    score = sum(f["score"] for f in filas) / len(filas)
    inval = "-"
    if convs:
        convs = [c for c in convs if c["id"] in ev]
        turnos = sum(c.get("turnos", 0) for c in convs)
        inval = f"{100 * sum(c.get('invalidos', 0) for c in convs) / max(turnos, 1):.1f}"
    pct = []
    for c in criterios:
        vs = [f["checklist"][c] for f in filas if (f.get("checklist") or {}).get(c) is not None]
        pct.append(f"{100 * sum(vs) / len(vs):.0f}" if vs else "-")
    filas_csv.append([nombre, len(filas), f"{score:.3f}", inval] + pct)

ancho = [max(len(str(x)) for x in col) for col in zip(hdr, *filas_csv)]
corto = ["variante", "n", "score", "inval%", "no_revela", "no_calcula", "valida", "aritm", "cierra", "evita_fallo"]
ancho = [max(a, len(c)) for a, c in zip(ancho, corto)]
print(" | ".join(c.ljust(a) for c, a in zip(corto, ancho)))
for f in filas_csv:
    print(" | ".join(str(x).ljust(a) for x, a in zip(f, ancho)))
with open(here / "resultados_pedagogicos.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(filas_csv)
print(f"\nGuardado en {here / 'resultados_pedagogicos.csv'}  (criterios en % de casos donde aplica)")
PY
}

if [ "${1:-}" = "--resumen" ]; then resumen; exit 0; fi
VARS=("$@"); [ ${#VARS[@]} -eq 0 ] && VARS=(F16 Q4_K_M-imat Q4_0-imat)
command -v curl >/dev/null || { echo "Falta curl"; exit 1; }
[ -x "$BIN/llama-server" ] || { echo "Falta llama-server en $BIN"; exit 1; }
grep -q '^OPENROUTER_API_KEY=' "$PROJ/.env" || { echo "Falta OPENROUTER_API_KEY en $PROJ/.env (lo usa el juez)"; exit 1; }

for v in "${VARS[@]}"; do correr_variante "$v"; done
resumen
