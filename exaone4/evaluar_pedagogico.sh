#!/usr/bin/env bash
# =============================================================================
#  Lumi · Fase 2 · Evaluación pedagógica de los GGUF de EXAONE con el juez
#
#  Para cada variante: levanta llama-server con el GGUF, hace una prueba de humo,
#  reproduce los 60 guiones de evaluación (data/particion.json, los que NO se usaron
#  para la imatrix) con tools/correr_guiones.py (mismo prompt y parámetros que
#  NubeEngine) y los califica con tools/juez.py (GPT-5 por OpenRouter).
#  Compara contra la nube (lumi_project/evaluacion_calibracion.jsonl).
#
#  Uso (desde model_quantization_phase2/exaone4/):
#     bash evaluar_pedagogico.sh                              # F16 Q8_0 Q4_K_M-imat Q4_0-imat
#     bash evaluar_pedagogico.sh F16 Q5_K_M-imat Q4_0-imat    # las que quieras
#     bash evaluar_pedagogico.sh todas                        # F16 + las 7 variantes
#     bash evaluar_pedagogico.sh --resumen                    # solo la tabla final
#     FORMATO=json_object bash evaluar_pedagogico.sh F16      # sin decodificación restringida
#
#  FORMATO (default: schema)
#    schema       el servidor restringe la salida al esquema JSON de la app (claves guia,
#                 opciones, pregunta_cierre). Es como correría on-device con llama.cpp.
#    json_object  igual que NubeEngine en la nube: solo obliga a que sea JSON. Con EXAONE
#                 1.2B esto dio 163/163 turnos inválidos (inventaba claves como "tipo").
#  Los resultados de cada formato se guardan por separado en resultados_pedagogicos/<formato>/.
#
#  Es reanudable: los guiones y calificaciones ya hechos se omiten.
#  Costo del juez: ~60 llamadas a GPT-5 por variante.
# =============================================================================
set -euo pipefail

NAME="exaone4-1.2b"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"                  # lumi_project/
BIN="$HERE/work/llama.cpp/build/bin"
OUT="$HERE/work/gguf"
FORMATO="${FORMATO:-schema}"
RES="$HERE/resultados_pedagogicos/$FORMATO"
LOGS="$HERE/logs"
PORT="${PORT:-8091}"
SLOTS="${SLOTS:-4}"                 # conversaciones en paralelo
JUEZ_WORKERS="${JUEZ_WORKERS:-6}"
PY="${PY:-python3}"
TODAS=(F16 Q8_0 Q6_K Q5_K_M-imat Q4_K_M Q4_K_M-imat Q4_0 Q4_0-imat)
mkdir -p "$RES" "$LOGS"
log(){ printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }

case "$FORMATO" in schema|json_object) ;; *) echo "FORMATO debe ser schema o json_object"; exit 1;; esac

SERVER_PID=""
parar_servidor(){ if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; SERVER_PID=""; }
trap parar_servidor EXIT INT TERM

[ -f "$HERE/data/particion.json" ] || { echo "Falta data/particion.json: corre primero bash cuantizar_exaone.sh"; exit 1; }
EVAL_IDS=$("$PY" -c "import json;print(' '.join(json.load(open('$HERE/data/particion.json'))['eval']))")
N_EVAL=$(wc -w <<<"$EVAL_IDS" | tr -d ' ')
export LUMI_TUTOR_URL="http://127.0.0.1:$PORT/v1/chat/completions"
export LUMI_TUTOR_TIMEOUT=300
export LUMI_TUTOR_FORMATO="$FORMATO"

levantar_servidor(){ # $1 = variante
  local modelo="$OUT/$NAME-$1.gguf"
  [ -f "$modelo" ] || { echo "No existe $modelo (corre primero cuantizar_exaone.sh)"; exit 1; }
  log "Levantando llama-server con $1"
  # La plantilla embebida en el GGUF usa {%- continue %} y el motor Jinja de llama.cpp descarta
  # el mensaje de sistema; esta copia es idéntica pero sin continue.
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

prueba_humo(){ # una pregunta: confirma que el servidor responde y que el JSON tiene las claves de la app
  ( cd "$PROJ" && "$PY" - "$NAME-$1" <<'PY'
import sys
sys.path.insert(0, "tools")
from evaluar import Tutor
r = Tutor(sys.argv[1]).responder("¿Cuánto es 3 + 4?")
if r["parsed"] is None:
    print(f"⚠  Prueba de humo: formato inválido ({r['motivo']}). Respuesta:\n{r['raw'][:400]}")
else:
    print(f"Prueba de humo OK ({r['segundos']} s): {r['parsed']['guia'][:120]}")
PY
  ) || { echo "La prueba de humo falló; revisa $LOGS/server_$1.log"; exit 1; }
}

correr_variante(){ # $1 = variante
  local v="$1" guiones="$RES/guiones_$1.json" juez="$RES/juez_$1.jsonl"
  local hechos=0
  [ -f "$guiones" ] && hechos=$("$PY" -c "import json;print(len(json.load(open('$guiones'))))" 2>/dev/null || echo 0)

  if [ "$hechos" -lt "$N_EVAL" ]; then
    levantar_servidor "$v"
    prueba_humo "$v"
    log "Reproduciendo $N_EVAL guiones con $v · formato $FORMATO ($hechos ya hechos)"
    local ids=(); for i in $EVAL_IDS; do ids+=(--id "$i"); done
    ( cd "$PROJ" && "$PY" tools/correr_guiones.py guiones.json "${ids[@]}" \
        --modelo "$NAME-$v" --workers "$SLOTS" -o "$guiones" ) || true
    parar_servidor
    hechos=$("$PY" -c "import json;print(len(json.load(open('$guiones'))))" 2>/dev/null || echo 0)
  else
    echo "$v: guiones completos ($hechos/$N_EVAL)"
  fi
  [ "$hechos" -gt 0 ] || { echo "$v: no hay guiones para calificar"; return; }

  local calificados=0
  [ -f "$juez" ] && calificados=$(grep -c '"score"' "$juez" || true)
  if [ "$calificados" -lt "$hechos" ]; then
    log "Calificando $v con el juez (GPT-5)"
    # el juez sí va a OpenRouter: se quita la URL local solo para este paso
    ( cd "$PROJ" && env -u LUMI_TUTOR_URL -u LUMI_TUTOR_FORMATO \
        "$PY" tools/juez.py "$guiones" --workers "$JUEZ_WORKERS" -o "$juez" ) | tail -20
  else
    echo "$v: ya calificada ($calificados)"
  fi
}

resumen(){
  log "Resumen"
  "$PY" - "$HERE" "$PROJ" "$HERE/resultados_pedagogicos" <<'PY'
import json, sys, csv, pathlib
here, proj, base = map(pathlib.Path, sys.argv[1:4])
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

fuentes = {}
nube = proj / "evaluacion_calibracion.jsonl"
if nube.exists():
    fuentes["nube (Qwen3-VL-235B)"] = (leer_jsonl(nube), None)
for carpeta in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
    for j in sorted(carpeta.glob("juez_*.jsonl")):
        v = j.stem[len("juez_"):]
        g = carpeta / f"guiones_{v}.json"
        fuentes[f"{v} [{carpeta.name}]"] = (leer_jsonl(j), json.load(open(g)) if g.exists() else None)

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

if not filas_csv:
    print("Todavía no hay calificaciones."); sys.exit()
corto = ["variante", "n", "score", "inval%", "no_revela", "no_calcula", "valida", "aritm", "cierra", "evita_fallo"]
ancho = [max(len(c), *(len(str(f[i])) for f in filas_csv)) for i, c in enumerate(corto)]
print(" | ".join(c.ljust(a) for c, a in zip(corto, ancho)))
for f in filas_csv:
    print(" | ".join(str(x).ljust(a) for x, a in zip(f, ancho)))
with open(here / "resultados_pedagogicos.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(hdr); w.writerows(filas_csv)
print(f"\nGuardado en {here / 'resultados_pedagogicos.csv'}  (criterios en % de casos donde aplica)")
PY
}

if [ "${1:-}" = "--resumen" ]; then resumen; exit 0; fi
VARS=("$@")
[ ${#VARS[@]} -eq 0 ] && VARS=(F16 Q8_0 Q4_K_M-imat Q4_0-imat)
[ "${VARS[0]}" = "todas" ] && VARS=("${TODAS[@]}")
command -v curl >/dev/null || { echo "Falta curl"; exit 1; }
[ -x "$BIN/llama-server" ] || { echo "Falta llama-server en $BIN (corre primero cuantizar_exaone.sh)"; exit 1; }
grep -q '^OPENROUTER_API_KEY=.' "$PROJ/.env" || { echo "Falta OPENROUTER_API_KEY en $PROJ/.env (lo usa el juez)"; exit 1; }

for v in "${VARS[@]}"; do correr_variante "$v"; done
resumen
