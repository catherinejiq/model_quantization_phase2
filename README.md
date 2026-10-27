# model_quantization_phase2

Cuantización de SLMs para Lumi (fase 2: SLM cuantizado + RAG, ejecución en teléfono con llama.cpp).

## EXAONE 4.0 1.2B

```bash
brew install cmake                 # si no lo tienes
bash cuantizar_exaone.sh           # pipeline completo
bash cuantizar_exaone.sh eval      # re-evaluar
bash cuantizar_exaone.sh limpiar   # liberar disco
```

| Paso | Qué hace | Salida |
|---|---|---|
| 1 | Descarga `LGAI-EXAONE/EXAONE-4.0-1.2B` y convierte a GGUF F16 | `work/gguf/*-F16.gguf` |
| 2 | Renderiza `../calibracion.json` con el system prompt de `NubeEngine` y la plantilla de chat de EXAONE; partición 80/20 estratificada por categoría (semilla 42) | `data/calib.txt`, `data/eval.txt`, `data/particion.json` |
| 3 | `llama-imatrix` sobre las conversaciones de calibración | `work/gguf/*.imatrix` |
| 4 | Cuantiza Q4_K_M y Q4_0, con y sin imatrix | `work/gguf/*.gguf` |
| 5 | PPL y divergencia KL vs F16 en conversaciones no vistas + velocidad en CPU | `resultados_exaone.csv`, `logs/` |

Requisitos: macOS Apple Silicon, Xcode Command Line Tools, cmake, Python 3.10+, ~11 GB libres.
La versión exacta de llama.cpp usada queda en `logs/llama_cpp_version.txt`.

Licencia del modelo: EXAONE AI Model License 1.2 – NC (uso de investigación/educativo, no comercial).
