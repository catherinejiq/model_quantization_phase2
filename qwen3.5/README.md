# Qwen3.5-2B · Fase 2

Mismo protocolo que `../exaone4/` (calibración, partición, renderizado, variantes y métricas),
para comparar ambos modelos en igualdad de condiciones.

```bash
bash cuantizar_qwen.sh                          # 1) cuantiza y evalúa PPL/KLD/velocidad
bash evaluar_pedagogico.sh                      # 2) evaluación pedagógica con el juez
FORMATO=json_object bash evaluar_pedagogico.sh F16   # control sin decodificación restringida
```

- Solo se exporta el modelo de texto (Qwen3.5-2B es de visión+lenguaje; Lumi no usa imágenes).
- Si existe `../exaone4/work/llama.cpp`, se reutiliza: ambos modelos quedan con el mismo commit.
- Disco: ~22 GB libres (el `.kld` de referencia pesa ~4 GB por el vocabulario de 248k tokens).
- Renderizado v2: el system prompt va en 1 conversación por categoría en `calib.txt` y en ninguna de `eval.txt`
  (antes era el 85 % del texto). La evaluación pedagógica sí usa el prompt completo.
- Resultados: `resultados_qwen.csv`, `resultados_pedagogicos.csv`, `logs/`.
- Licencia del modelo: Apache 2.0.
