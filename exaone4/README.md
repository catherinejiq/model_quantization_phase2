# EXAONE 4.0 1.2B · Fase 2

Cuantización con llama.cpp y evaluación (estadística + pedagógica) para Lumi.
Calibración: `lumi_project/calibracion_corregido.json` (300 conversaciones con respuestas corregidas a mano).

```bash
brew install cmake                              # si no lo tienes
bash cuantizar_exaone.sh                        # 1) cuantiza y evalúa PPL/KLD/velocidad
bash evaluar_pedagogico.sh                      # 2) evaluación pedagógica con el juez
bash evaluar_pedagogico.sh todas                #    o todas las variantes
FORMATO=json_object bash evaluar_pedagogico.sh F16   # control sin decodificación restringida
```

| Paso | Qué hace | Salida |
|---|---|---|
| 1 | Descarga `LGAI-EXAONE/EXAONE-4.0-1.2B` → GGUF F16 | `work/gguf/*-F16.gguf` |
| 2 | Renderiza la calibración con el system prompt de `NubeEngine` y la plantilla de EXAONE; partición 80/20 estratificada (semilla 42) | `data/calib.txt`, `data/eval.txt`, `data/particion.json`, `data/fuente.txt` |
| 3 | `llama-imatrix` sobre las 240 conversaciones de calibración | `work/gguf/*.imatrix` |
| 4 | Variantes Q8_0 · Q6_K · Q5_K_M-imat · Q4_K_M · Q4_K_M-imat · Q4_0 · Q4_0-imat | `work/gguf/*.gguf` |
| 5 | PPL y KLD vs F16 en las 60 conversaciones no vistas + velocidad en CPU | `resultados_exaone.csv`, `logs/` |
| 6 | 60 guiones por variante en `llama-server` + juez GPT-5 | `resultados_pedagogicos/<formato>/`, `resultados_pedagogicos.csv` |

Notas
- La partición es idéntica a la de `calibracion.json` (mismos ids), así que la comparación con la nube sigue siendo válida.
- `data/fuente.txt` guarda el SHA-256 del archivo de calibración: si cambia, el script rehace imatrix, variantes `-imat` y evaluación.
- Formato `schema`: llama-server restringe la salida al esquema JSON de la app. En la corrida anterior con `json_object`, EXAONE inventó claves propias en 163/163 turnos.
- Renderizado v2: el system prompt va en 1 conversación por categoría en `calib.txt` y en ninguna de `eval.txt` (antes era el 85 % del texto, así que la imatrix y la PPL/KLD medían sobre todo el prompt). Al volver a correr el script se rehacen solo los datos, la imatrix, las variantes `-imat` y la evaluación.
- Versión de llama.cpp usada: `logs/llama_cpp_version.txt`.
- Licencia del modelo: EXAONE AI Model License 1.2 – NC (investigación/educación, no comercial).
