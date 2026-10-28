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
- Renderizado v3: en `calib.txt` todas las conversaciones llevan el system prompt y la imatrix usa contexto de 4096 tokens, así cada respuesta se calibra con su prompt delante (v1 usaba 512 y el prompt nunca quedaba junto a la respuesta). `eval.txt` va sin prompt para que la PPL/KLD mida el diálogo; el seguimiento del prompt se mide en la evaluación pedagógica.
- Resultados: `resultados_qwen.csv`, `resultados_pedagogicos.csv`, `logs/`.
- Licencia del modelo: Apache 2.0.
