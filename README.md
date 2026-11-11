# Caso 1 (Regresión): consumo de combustible del CPS Tritón

Proyecto integrador de **Estadística Aplicada** (Ingeniería Industrial, Universidad de Santiago de Chile, 2026).
Profesor: Luis Rojo-González, Ph.D.

**Integrantes:** Sylvia Montanares, Vladimir Painiqueo, Isidora Rioseco, Hernán Silva y Germán Troncoso.

## ¿Qué se predice?

El consumo instantáneo de combustible (kg/s) del crucero CPS Tritón, en intervalos de 5 minutos, a partir de su
velocidad y de las condiciones del clima y el mar. Es un problema de regresión (no un pronóstico de series de tiempo).
Se comparan 12 modelos (baseline, OLS, Ridge, Lasso, Elastic-net, Splines/GAM, árbol, Random Forest, Boosting,
Averaging y Stacking) con partición temporal, validación cruzada por bloques y un test evaluado una sola vez.
**Modelo final: Splines/GAM.**

## Estructura del repositorio

```
.
├── README.md                  <- este archivo
├── codigo/                    <- script reproducible en R (partes 0 a 18)
├── datos/                     <- datos del subconjunto cps_triton y enlace_descarga.txt
├── resultados/                <- archivos generados por la corrida final del script
├── informe/                   <- informe final (PDF)
└── presentacion/              <- presentación (PPTX)
```

## Datos

Dataset **FuelCast** (subconjunto `cps_triton`), Hugging Face: https://huggingface.co/datasets/krohnedigital/FuelCast
Licencia CC BY-NC-ND 4.0 (uso no comercial). Detalles en `datos/enlace_descarga.txt`.
El script lee el archivo `CPS_Triton.parquet`, que debe estar en la carpeta desde la que se ejecuta el script.

## Cómo reproducir los resultados

1. Abrir `codigo/Caso1_CPS_Triton_Reproducible_Grupo4.ipynb` en Google Colab (*Archivo > Subir cuaderno*) con el entorno de ejecución de **R** (*Entorno de ejecución > Cambiar tipo de entorno de ejecución > R*), en una **sesión limpia**. También se puede ejecutar el código en R local (versión 4.6.1 en la corrida final).
2. Subir `CPS_Triton.parquet` (carpeta `datos/`) al panel de archivos de Colab, junto al cuaderno (o ajustar la ruta en la Parte 1).
3. Ejecutar el script completo de principio a fin (partes 0 a 18). Instala los paquetes que falten:
   `glmnet`, `mgcv`, `rpart`, `ranger`, `gbm`, `nnls`, `dplyr`, `ggplot2` y `arrow` (versiones exactas en `resultados/configuracion_y_sesion.txt`).
4. Los resultados quedan en la carpeta `resultados/` (tablas de validación y test, incertidumbre, coeficientes,
   grillas de validación cruzada y `configuracion_y_sesion.txt` con las versiones usadas).

**Notas de reproducibilidad**

- Semilla global: 123. Todo el código usa un solo hilo para que los resultados no cambien entre ejecuciones.
- El test se evalúa una sola vez (Parte 17), con el modelo final fijado antes de verlo (`modelo_final_manual`).
- El script se detiene si hay columnas `Consumer_*` o `Propeller_*` como predictoras, o si los bloques de
  entrenamiento, validación y test se solapan o no respetan el orden del tiempo.

## Resultados principales (test, evaluado una sola vez)

| Métrica | Baseline (promedio por zona) | GAM (modelo final) |
|---|---|---|
| RMSE general (kg/s) | 0.0312 | 0.0254 |
| RMSE subgrupo 1 a 6 m/s (kg/s) | 0.0427 | 0.0404 |
| MAE (kg/s) | 0.0245 | 0.0194 |
| R² | 0.708 | 0.806 |

El GAM reduce el RMSE general en 18.6 % frente al baseline; en el subgrupo de 1 a 6 m/s la mejora es de 5.4 %.
