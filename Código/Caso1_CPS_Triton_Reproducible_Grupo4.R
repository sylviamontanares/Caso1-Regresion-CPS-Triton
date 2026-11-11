# CASO 1 (REGRESIÓN) - CONSUMO DE COMBUSTIBLE DEL CPS TRITON
# SCRIPT COMPLETO: de los datos al test final, de principio a fin.
# CASO 1 (REGRESIÓN): CONSUMO DE COMBUSTIBLE DEL CPS TRITON
# Sistema completo: baseline, OLS, Ridge/Lasso/Elastic-net, splines/GAM,
# árbol, Random Forest, boosting y combinación (averaging y stacking)

# PARTE 0: CONFIGURACIÓN, PAQUETES Y SEMILLA

rm(list = ls())   # limpia el entorno: nada de una ejecución anterior contamina la siguiente
RNGkind(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")  # generador fijo (no depende de la versión de R)

semilla           <- 123                 # semilla global (reproducibilidad)
prop_train        <- 0.60                # 60 % de las filas para entrenar
prop_valid        <- 0.20                # 20 % para validar (el resto es test)
hueco             <- 60                  # filas que se descartan entre bloques
n_bloques_cv      <- 5                   # bloques de tiempo para validar (CV y métricas)
subgrupo          <- "intermedia (1 a 6)"  # subgrupo crítico a reportar
umbral_relevancia <- 0.01                # efecto relevante: >= 1 % del consumo medio
n_arboles_rf      <- 300                 # árboles de cada Random Forest
nivel_intervalo   <- 0.95                # nivel de los intervalos de predicción
modelo_final_manual <- "Splines / GAM"  # modelo final acordado ANTES de ver el test
                                         # (coincide con la regla automática de la PARTE 15)
EJECUTAR_TEST     <- TRUE                # TRUE = la PARTE 17 evalúa el test (UNA vez, al final)
archivo_csv       <- "CPS_Triton.csv"
archivo_parquet   <- "CPS_Triton.parquet"
carpeta_salida    <- "resultados"
n_hilos           <- 1                   # 1 hilo en BLAS, ranger y mgcv: el paralelismo cambia el orden de las sumas
                                         # y puede alterar los últimos decimales (y con ello, a veces, el modelo elegido)

# Paquetes (se instalan solo si faltan):
#   glmnet = Ridge, Lasso y Elastic-net      mgcv   = splines / GAM
#   rpart  = árbol de regresión              ranger = Random Forest (rápido)
#   gbm    = Gradient Boosting               nnls   = pesos del stacking (>= 0)
#   dplyr  = manipulación de datos           ggplot2 = gráficos avanzados (DataCamp)
#   arrow  = para leer archivos parquet
paquetes <- c("glmnet", "mgcv", "rpart", "ranger", "gbm", "nnls", "dplyr", "ggplot2", "arrow")
for (p in paquetes) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
invisible(lapply(paquetes, library, character.only = TRUE))

# BLAS/OpenMP de un solo hilo (opcional: si el paquete no se puede instalar, se omite sin error)
if (!requireNamespace("RhpcBLASctl", quietly = TRUE)) try(install.packages("RhpcBLASctl"), silent = TRUE)
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
  RhpcBLASctl::blas_set_num_threads(n_hilos)
  RhpcBLASctl::omp_set_num_threads(n_hilos)
}

set.seed(semilla)   # fija el azar: los resultados salen siempre iguales


# PARTE 1: LEER Y REVISAR LOS DATOS
# Cambiado para leer desde el archivo parquet disponible
d <- arrow::read_parquet("CPS_Triton.parquet")

dim(d)                              # filas y columnas (esperado: 25351 y 62)
faltantes <- colSums(is.na(d))
faltantes[faltantes > 0]            # columnas con faltantes (NA) y cuántos
is.unsorted(d$index, na.rm = TRUE)  # esperado FALSE: filas en orden de tiempo


# PARTE 2: ARMAR LA TABLA DE VARIABLES (x)

col_respuesta <- "Consumer_Total_MomentaryFuel"
cols_pred_origen <- c("Ship_SpeedOverGround", "Environment_SeaFloorDepth",
                      "Weather_Temperature2M", "Weather_WindSpeed10M",
                      "Weather_WindDirection10M", "Ship_Heading",
                      "Weather_WaveHeight", "Weather_WavePeriod",
                      "Weather_WaveDirection", "Weather_OceanCurrentVelocity",
                      "Weather_OceanCurrentDirection")

# VERIFICACIONES ANTI-FUGA
stopifnot(all(c(col_respuesta, cols_pred_origen) %in% names(d)))
stopifnot(!any(grepl("^(Consumer_|Propeller_)", cols_pred_origen)))

rad <- pi / 180
rel <- function(dir, rumbo) (dir - rumbo) * rad

x_completa <- data.frame(
  consumo       = d[[col_respuesta]],              # lo que queremos predecir
  velocidad     = d$Ship_SpeedOverGround,          # velocidad (m/s)
  profundidad   = d$Environment_SeaFloorDepth,
  temp          = d$Weather_Temperature2M,
  viento_vel    = d$Weather_WindSpeed10M,
  viento_frente = cos(rel(d$Weather_WindDirection10M, d$Ship_Heading)),
  viento_lado   = sin(rel(d$Weather_WindDirection10M, d$Ship_Heading)),
  ola_alt       = d$Weather_WaveHeight,
  ola_per       = d$Weather_WavePeriod,
  ola_frente    = cos(rel(d$Weather_WaveDirection, d$Ship_Heading)),
  ola_lado      = sin(rel(d$Weather_WaveDirection, d$Ship_Heading)),
  corr_vel      = d$Weather_OceanCurrentVelocity,
  corr_frente   = cos(rel(d$Weather_OceanCurrentDirection, d$Ship_Heading)),
  corr_lado     = sin(rel(d$Weather_OceanCurrentDirection, d$Ship_Heading))
)

completas <- complete.cases(x_completa)
x         <- x_completa[completas, ]
fila_orig <- which(completas)

dim(x)        # esperado: ~25308 filas y 14 columnas
summary(x)


# PARTE 3: PARTICIÓN EN TIEMPO (ENTRENAMIENTO / VALIDACIÓN / TEST)

n  <- nrow(x)
i1 <- floor(prop_train * n)
i2 <- floor((prop_train + prop_valid) * n)

idx_train <- 1:i1
idx_valid <- (i1 + hueco + 1):i2
idx_test  <- (i2 + hueco + 1):n

train <- x[idx_train, ]
valid <- x[idx_valid, ]
test  <- x[idx_test, ]
c(train = nrow(train), valid = nrow(valid), test = nrow(test))

# VERIFICACIONES ANTI-FUGA TEMPORAL
stopifnot(!is.unsorted(fila_orig))
stopifnot(max(idx_train) < min(idx_valid), max(idx_valid) < min(idx_test))
stopifnot(length(intersect(idx_train, idx_valid)) == 0,
          length(intersect(idx_valid, idx_test))  == 0,
          length(intersect(idx_train, idx_test))  == 0)
stopifnot(min(idx_valid) - max(idx_train) > hueco,
          min(idx_test)  - max(idx_valid) > hueco)

rmse <- function(real, pred) sqrt(mean((real - pred)^2))
metricas <- function(real, pred) {
  error <- real - pred
  c(MAE  = mean(abs(error)),
    RMSE = sqrt(mean(error^2)),
    R2   = 1 - sum(error^2) / sum((real - mean(real))^2))
}

zona_vel <- function(v) cut(v, breaks = c(-Inf, 1, 6, Inf),
                            labels = c("casi detenido (<1)", "intermedia (1 a 6)", "crucero (>6)"))
zv <- zona_vel(valid$velocidad)


# EDA (ANÁLISIS EXPLORATORIO) - CONSUMO DE COMBUSTIBLE DEL CPS TRITON

xe <- train
vars_pred <- names(xe)[-1]

# E1: ¿CÓMO SE DISTRIBUYE EL CONSUMO?
summary(xe$consumo)
hist(xe$consumo, breaks = 60, col = "grey80",
     xlab = "Consumo total (kg/s)", main = "Distribución del consumo")
sum(xe$consumo == 0)

# Resumen agrupado por zonas de velocidad con dplyr
resumen_dplyr <- xe %>%
  mutate(zona = zona_vel(velocidad)) %>%
  group_by(zona) %>%
  summarise(
    consumo_medio = mean(consumo, na.rm = TRUE),
    consumo_sd    = sd(consumo, na.rm = TRUE),
    consumo_max   = max(consumo, na.rm = TRUE),
    casos         = n()
  ) %>%
  arrange(desc(consumo_medio))
print("--- Resumen por zonas de velocidad (dplyr) ---")
print(resumen_dplyr)

# Curva suave de consumo vs velocidad con ggplot2
g_eda <- ggplot(xe, aes(x = velocidad, y = consumo)) +
  geom_point(alpha = 0.2, color = "steelblue", size = 0.8) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "firebrick", fill = "pink") +
  theme_minimal(base_size = 12) +
  labs(title = "Relación Consumo vs Velocidad (Ajuste GAM en EDA)",
       subtitle = "Visualización implementada con ggplot2",
       x = "Velocidad sobre el fondo (m/s)",
       y = "Consumo total (kg/s)")
print(g_eda)

# E2: ¿CÓMO CAMBIA EN EL TIEMPO?
par(mfrow = c(2, 1))
plot(xe$consumo, type = "l", xlab = "Fila (tiempo)", ylab = "Consumo (kg/s)", main = "Consumo en el tiempo")
plot(xe$velocidad, type = "l", xlab = "Fila (tiempo)", ylab = "Velocidad (m/s)", main = "Velocidad en el tiempo")
par(mfrow = c(1, 1))

# E3: DISTRIBUCIÓN DE CADA VARIABLE PREDICTORA
par(mfrow = c(4, 4), mar = c(4, 4, 2, 1))
for (v in vars_pred) hist(xe[[v]], breaks = 40, col = "grey80", main = v, xlab = "")
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

# E4: RELACIÓN DE CADA VARIABLE CON EL CONSUMO
set.seed(semilla)
muestra <- xe[sample(nrow(xe), min(3000, nrow(xe))), ]

par(mfrow = c(4, 4), mar = c(4, 4, 2, 1))
for (v in vars_pred) {
  plot(muestra[[v]], muestra$consumo, pch = 16, cex = 0.3, main = v, xlab = "", ylab = "consumo")
}
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

cor_consumo <- data.frame(
  pearson  = cor(xe[, vars_pred], xe$consumo),
  spearman = cor(xe[, vars_pred], xe$consumo, method = "spearman")
)
round(cor_consumo[order(-abs(cor_consumo$spearman)), ], 2)

# E5: CORRELACIÓN ENTRE LAS PREDICTORAS (MULTICOLINEALIDAD)
cm <- cor(xe[, vars_pred])
cm[upper.tri(cm, diag = TRUE)] <- NA
alto <- which(abs(cm) > 0.7, arr.ind = TRUE)
data.frame(var1 = rownames(cm)[alto[, 1]], var2 = colnames(cm)[alto[, 2]], r = round(cm[alto], 2))

vif <- sapply(vars_pred, function(v) {
  r2 <- summary(lm(reformulate(setdiff(vars_pred, v), response = v), data = xe))$r.squared
  1 / (1 - r2)
})
round(sort(vif, decreasing = TRUE), 2)

# E6: VALORES ATÍPICOS
n_atipicos <- sapply(xe, function(z) {
  q <- quantile(z, c(0.25, 0.75))
  sum(z < q[1] - 1.5 * diff(q) | z > q[2] + 1.5 * diff(q))
})
n_atipicos[n_atipicos > 0]
boxplot(xe[, c("consumo", "velocidad", "viento_vel", "ola_alt", "corr_vel")], main = "Boxplot de variables seleccionadas", las = 2)

# E7: DEPENDENCIA ENTRE OBSERVACIONES
acf(xe$consumo, lag.max = 100, main = "Autocorrelación del consumo")
cor(xe$consumo[-1], xe$consumo[-nrow(xe)])


# PARTE 4: BASELINES Y SOLUCIÓN DE REFERENCIA (SOLO VELOCIDAD)

b_media <- rep(mean(train$consumo), nrow(valid))
medias_zona <- tapply(train$consumo, zona_vel(train$velocidad), mean)
b_zona      <- as.numeric(medias_zona[as.character(zv)])

m1 <- lm(consumo ~ velocidad,                      data = train)
m2 <- lm(consumo ~ poly(velocidad, 2, raw = TRUE), data = train)
m3 <- lm(consumo ~ poly(velocidad, 3, raw = TRUE), data = train)

rbind(baseline_media       = metricas(valid$consumo, b_media),
      baseline_prom_zona   = metricas(valid$consumo, b_zona),
      referencia_grado1    = metricas(valid$consumo, predict(m1, newdata = valid)),
      referencia_grado2    = metricas(valid$consumo, predict(m2, newdata = valid)),
      referencia_grado3    = metricas(valid$consumo, predict(m3, newdata = valid)))

plot(train$velocidad, train$consumo, pch = 16, cex = 0.3, xlab = "Velocidad (m/s)", ylab = "Consumo (kg/s)")
v <- seq(0, 9, length.out = 200)
lines(v, predict(m3, newdata = data.frame(velocidad = v)), col = "red", lwd = 2)



# PARTE 5: REGRESIÓN LINEAL MÚLTIPLE (OLS) CON CLIMA Y MAR

m_clima <- lm(consumo ~ poly(velocidad, 3, raw = TRUE) +
                profundidad + temp + viento_vel + viento_frente + viento_lado +
                ola_alt + ola_per + ola_frente + ola_lado +
                corr_vel + corr_frente + corr_lado,
              data = train)

p_ref   <- predict(m3,      newdata = valid)
p_clima <- predict(m_clima, newdata = valid)

rbind(referencia_cubica = metricas(valid$consumo, p_ref),
      ols_con_clima     = metricas(valid$consumo, p_clima))

rmse_por_zona <- function(pred) {
  sapply(levels(zv), function(z) rmse(valid$consumo[zv == z], pred[zv == z]))
}
rbind(baseline_prom_zona = rmse_por_zona(b_zona),
      referencia_cubica  = rmse_por_zona(p_ref),
      ols_con_clima      = rmse_por_zona(p_clima))
table(zv)


# PARTE 6: COEFICIENTES (SIGNIFICANCIA vs RELEVANCIA)

summary(m_clima)
confint(m_clima)

vars <- c("profundidad", "temp", "viento_vel", "viento_frente", "viento_lado",
          "ola_alt", "ola_per", "ola_frente", "ola_lado",
          "corr_vel", "corr_frente", "corr_lado")
train_std <- train
train_std[vars] <- scale(train[vars])

m_std <- lm(consumo ~ poly(velocidad, 3, raw = TRUE) +
              profundidad + temp + viento_vel + viento_frente + viento_lado +
              ola_alt + ola_per + ola_frente + ola_lado +
              corr_vel + corr_frente + corr_lado,
            data = train_std)

coef_std <- summary(m_std)$coefficients[vars, ]
ic_std   <- confint(m_std)[vars, ]
tab_relevancia <- data.frame(
  variable            = vars,
  efecto_por_sd       = coef_std[, "Estimate"],
  IC_inf              = ic_std[, 1],
  IC_sup              = ic_std[, 2],
  p_valor             = coef_std[, "Pr(>|t|)"],
  efecto_pct_consumo  = 100 * coef_std[, "Estimate"] / mean(train$consumo),
  significativa       = coef_std[, "Pr(>|t|)"] < 0.05,
  relevante           = abs(coef_std[, "Estimate"]) / mean(train$consumo) >= umbral_relevancia,
  row.names = NULL
)
tab_relevancia <- tab_relevancia[order(-abs(tab_relevancia$efecto_pct_consumo)), ]
print(tab_relevancia, digits = 3)


# PARTE 7: DIAGNÓSTICO DE LOS RESIDUOS

res <- resid(m_clima)
aj  <- fitted(m_clima)

par(mfrow = c(1, 3))
plot(aj, res, pch = 16, cex = 0.3, xlab = "Predicción", ylab = "Residuo", main = "Residuos vs predicción")
abline(h = 0, col = "red")
qqnorm(res, pch = 16, cex = 0.3, main = "Normalidad de residuos")
qqline(res, col = "red")
acf(res, lag.max = 60, main = "Autocorrelación de residuos")
par(mfrow = c(1, 1))

cor(res[-1], res[-length(res)])


# PARTE 8: RIDGE, LASSO Y ELASTIC-NET

formula_x <- ~ velocidad + I(velocidad^2) + I(velocidad^3) +
  profundidad + temp + viento_vel + viento_frente + viento_lado +
  ola_alt + ola_per + ola_frente + ola_lado +
  corr_vel + corr_frente + corr_lado

stopifnot(!("consumo" %in% all.vars(formula_x)))

X_train <- model.matrix(formula_x, train)[, -1]
X_valid <- model.matrix(formula_x, valid)[, -1]
y_train <- train$consumo

bloques <- cut(seq_len(nrow(train)), breaks = n_bloques_cv, labels = FALSE)
stopifnot(length(bloques) == nrow(X_train))

cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0,   foldid = bloques)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1,   foldid = bloques)
cv_enet  <- cv.glmnet(X_train, y_train, alpha = 0.5, foldid = bloques)

pred_glm <- function(cv) as.numeric(predict(cv, newx = X_valid, s = "lambda.1se"))

rbind(ridge       = metricas(valid$consumo, pred_glm(cv_ridge)),
      lasso       = metricas(valid$consumo, pred_glm(cv_lasso)),
      elastic_net = metricas(valid$consumo, pred_glm(cv_enet)))

coef(cv_lasso, s = "lambda.1se")
plot(cv_lasso)

comparar_lambda <- function(cv) {
  p_min <- as.numeric(predict(cv, newx = X_valid, s = "lambda.min"))
  c(lambda_min      = cv$lambda.min,
    lambda_1se      = cv$lambda.1se,
    RMSE_cv_min     = sqrt(cv$cvm[cv$lambda == cv$lambda.min]),
    RMSE_cv_1se     = sqrt(cv$cvm[cv$lambda == cv$lambda.1se]),
    RMSE_valid_min  = rmse(valid$consumo, p_min),
    RMSE_valid_1se  = rmse(valid$consumo, pred_glm(cv)))
}
signif(rbind(ridge       = comparar_lambda(cv_ridge),
             lasso       = comparar_lambda(cv_lasso),
             elastic_net = comparar_lambda(cv_enet)), 4)


# PARTE 9: VALIDACIÓN CRUZADA POR BLOQUES (HERRAMIENTA COMÚN)

cv_bloques <- function(ajustar_y_predecir, datos = train) {
  stopifnot(nrow(datos) == length(bloques))
  pos  <- seq_len(nrow(datos))
  pred <- NULL
  for (b in sort(unique(bloques))) {
    en_b  <- which(bloques == b)
    lejos <- which(pos < min(en_b) - hueco | pos > max(en_b) + hueco)
    stopifnot(length(intersect(lejos, en_b)) == 0)
    p <- as.matrix(ajustar_y_predecir(datos[lejos, ], datos[en_b, ]))
    if (is.null(pred)) pred <- matrix(NA_real_, nrow(datos), ncol(p),
                                      dimnames = list(NULL, colnames(p)))
    pred[en_b, ] <- p
  }
  if (ncol(pred) == 1) as.numeric(pred) else pred
}
rmse_cv <- function(pred_oof) rmse(train$consumo, pred_oof)

afinar <- function(grilla, crear_modelo, minimos = NULL) {
  oof <- vector("list", nrow(grilla))
  for (k in seq_len(nrow(grilla))) {
    oof[[k]] <- cv_bloques(crear_modelo(grilla[k, , drop = FALSE]))
  }
  grilla$RMSE_cv <- sapply(oof, rmse_cv)
  mejor <- which.min(grilla$RMSE_cv)
  avisar_borde(grilla, grilla[mejor, , drop = FALSE], minimos)
  list(grilla = grilla, mejor = grilla[mejor, , drop = FALSE], oof = oof[[mejor]])
}

avisar_borde <- function(grilla, mejor, minimos = NULL, tolerancia = 0.01) {
  for (v in setdiff(names(grilla), "RMSE_cv")) {
    valores <- sort(unique(grilla[[v]]))
    if (v %in% names(minimos) && mejor[[v]] == minimos[[v]]) {
      message("Nota: '", v, "' = ", mejor[[v]], " es el mínimo posible de esa perilla.")
    } else if (length(valores) > 1 && mejor[[v]] %in% range(valores)) {
      ganancia <- 1 - mejor$RMSE_cv / min(grilla$RMSE_cv[grilla[[v]] != mejor[[v]]])
      if (ganancia >= tolerancia) {
        message("OJO: '", v, "' = ", mejor[[v]], " quedó en el borde de la grilla (",
                paste(valores, collapse = ", "), ") y mejora ", round(100 * ganancia, 1),
                " % al resto. Considera ampliarla.")
      } else {
        message("Nota: '", v, "' = ", mejor[[v]], " está en el borde, pero solo mejora ",
                round(100 * ganancia, 1), " % al resto (meseta): no hace falta ampliar.")
      }
    }
  }
}

vars_arbol <- setdiff(names(train), "consumo")
stopifnot(!("consumo" %in% vars_arbol))

predecir_con <- list(
  "Baseline (promedio por zona)"  = function(nd) as.numeric(medias_zona[as.character(zona_vel(nd$velocidad))]),
  "Referencia cubica (velocidad)" = function(nd) as.numeric(predict(m3,      newdata = nd)),
  "OLS con clima"                 = function(nd) as.numeric(predict(m_clima, newdata = nd)),
  "Ridge"       = function(nd) as.numeric(predict(cv_ridge, newx = model.matrix(formula_x, nd)[, -1], s = "lambda.1se")),
  "Lasso"       = function(nd) as.numeric(predict(cv_lasso, newx = model.matrix(formula_x, nd)[, -1], s = "lambda.1se")),
  "Elastic-net" = function(nd) as.numeric(predict(cv_enet,  newx = model.matrix(formula_x, nd)[, -1], s = "lambda.1se"))
)

stopifnot(isTRUE(all.equal(predecir_con[["Baseline (promedio por zona)"]](valid), b_zona)),
          isTRUE(all.equal(predecir_con[["OLS con clima"]](valid), as.numeric(p_clima))),
          isTRUE(all.equal(predecir_con[["Lasso"]](valid), pred_glm(cv_lasso))))

oof_ols <- cv_bloques(function(tr, te) predict(lm(formula(m_clima), data = tr), newdata = te))
c(RMSE_cv_OLS = rmse_cv(oof_ols))


# PARTE 10: MODELO FLEXIBLE - SPLINES / GAM

set.seed(semilla)

formula_gam <- function(k_vel, k_otras) {
  as.formula(paste0(
    "consumo ~ s(velocidad, k = ", k_vel, ")",
    " + s(profundidad, k = ", k_otras, ") + s(temp, k = ", k_otras, ")",
    " + s(viento_vel, k = ", k_otras, ") + s(ola_alt, k = ", k_otras, ")",
    " + s(ola_per, k = ", k_otras, ") + s(corr_vel, k = ", k_otras, ")",
    " + viento_frente + viento_lado + ola_frente + ola_lado + corr_frente + corr_lado"))
}

ajustar_gam <- function(datos, k_vel, k_otras) {
  bam(formula_gam(k_vel, k_otras), data = datos, method = "fREML", discrete = TRUE, nthreads = n_hilos)
}

grilla_gam <- expand.grid(k_vel = c(10, 20, 40, 80), k_otras = c(3, 4, 6))
res_gam <- afinar(grilla_gam, function(par) function(tr, te)
  predict(ajustar_gam(tr, par$k_vel, par$k_otras), newdata = te),
  minimos = c(k_vel = 3, k_otras = 3))
print(res_gam$grilla[order(res_gam$grilla$RMSE_cv), ], digits = 4)

m_gam <- ajustar_gam(train, res_gam$mejor$k_vel, res_gam$mejor$k_otras)
predecir_con[["Splines / GAM"]] <- function(nd) as.numeric(predict(m_gam, newdata = nd))
oof_gam <- res_gam$oof

summary(m_gam)$s.table
rbind(GAM_validacion = metricas(valid$consumo, predecir_con[["Splines / GAM"]](valid)))
plot(m_gam, pages = 1, shade = TRUE, scale = 0)


# PARTE 11: ÁRBOL DE REGRESIÓN

set.seed(semilla)

ajustar_arbol <- function(datos, cp, maxdepth) {
  rpart(consumo ~ ., data = datos[, c("consumo", vars_arbol)], method = "anova",
        control = rpart.control(cp = cp, maxdepth = maxdepth, minbucket = 20, xval = 0))
}

grilla_arbol <- expand.grid(cp = 0, maxdepth = c(3, 4, 5, 6, 7, 8, 10, 15))
res_arbol <- afinar(grilla_arbol, function(par) function(tr, te)
  predict(ajustar_arbol(tr, par$cp, par$maxdepth), newdata = te))
print(res_arbol$grilla[order(res_arbol$grilla$RMSE_cv), ], digits = 4)

m_arbol <- ajustar_arbol(train, res_arbol$mejor$cp, res_arbol$mejor$maxdepth)
predecir_con[["Arbol de regresion"]] <- function(nd) as.numeric(predict(m_arbol, newdata = nd))
oof_arbol <- res_arbol$oof

c(hojas = sum(m_arbol$frame$var == "<leaf>"))
rbind(arbol_validacion = metricas(valid$consumo, predecir_con[["Arbol de regresion"]](valid)))

arbol_corto <- ajustar_arbol(train, cp = 0.001, maxdepth = 3)
plot(arbol_corto, uniform = TRUE, margin = 0.08)
text(arbol_corto, use.n = TRUE, cex = 0.7)


# PARTE 12: RANDOM FOREST

set.seed(semilla)

ajustar_rf <- function(datos, mtry, min.node.size, importancia = "none") {
  ranger(consumo ~ ., data = datos[, c("consumo", vars_arbol)],
         num.trees = n_arboles_rf, mtry = mtry, min.node.size = min.node.size,
         importance = importancia, seed = semilla, num.threads = n_hilos)
}

grilla_rf <- expand.grid(mtry = c(6, 9, 12), min.node.size = c(50, 200, 500, 1000))
res_rf <- afinar(grilla_rf, function(par) function(tr, te)
  predict(ajustar_rf(tr, par$mtry, par$min.node.size), data = te, num.threads = n_hilos)$predictions)
print(res_rf$grilla[order(res_rf$grilla$RMSE_cv), ], digits = 4)

m_rf <- ajustar_rf(train, res_rf$mejor$mtry, res_rf$mejor$min.node.size, importancia = "permutation")
predecir_con[["Random Forest"]] <- function(nd) as.numeric(predict(m_rf, data = nd, num.threads = n_hilos)$predictions)
oof_rf <- res_rf$oof

rbind(RF_validacion = metricas(valid$consumo, predecir_con[["Random Forest"]](valid)))



# PARTE 13: GRADIENT BOOSTING

set.seed(semilla)

arboles_grid <- seq(100, 1000, by = 100)
ajustar_gbm <- function(datos, depth, shrink, n_arboles) {
  set.seed(semilla)
  gbm(consumo ~ ., data = datos[, c("consumo", vars_arbol)], distribution = "gaussian",
      n.trees = n_arboles, interaction.depth = depth, shrinkage = shrink,
      n.minobsinnode = 20, bag.fraction = 0.8, keep.data = FALSE, verbose = FALSE)
}

grilla_gbm <- expand.grid(depth = c(2, 4, 6), shrink = c(0.02, 0.05, 0.1))
res_gbm <- NULL
oof_gbm_lista <- list()
for (k in seq_len(nrow(grilla_gbm))) {
  par <- grilla_gbm[k, ]
  oof_k <- cv_bloques(function(tr, te) {
    m <- ajustar_gbm(tr, par$depth, par$shrink, max(arboles_grid))
    predict(m, newdata = te, n.trees = arboles_grid)
  })
  oof_gbm_lista[[k]] <- oof_k
  res_gbm <- rbind(res_gbm, data.frame(depth = par$depth, shrink = par$shrink,
                                       n_trees = arboles_grid,
                                       RMSE_cv = apply(oof_k, 2, rmse_cv)))
}
mejor_gbm <- res_gbm[which.min(res_gbm$RMSE_cv), ]
avisar_borde(res_gbm, mejor_gbm)
print(head(res_gbm[order(res_gbm$RMSE_cv), ], 8), digits = 4)
mejor_gbm

k_mejor <- which(grilla_gbm$depth == mejor_gbm$depth & grilla_gbm$shrink == mejor_gbm$shrink)
oof_gbm <- oof_gbm_lista[[k_mejor]][, which(arboles_grid == mejor_gbm$n_trees)]

con_mejor <- res_gbm$depth == mejor_gbm$depth & res_gbm$shrink == mejor_gbm$shrink
plot(res_gbm$n_trees[con_mejor], res_gbm$RMSE_cv[con_mejor], type = "b", pch = 16,
     xlab = "Número de árboles", ylab = "RMSE de CV (kg/s)",
     main = "Boosting: error de CV según n° de árboles")

m_gbm <- ajustar_gbm(train, mejor_gbm$depth, mejor_gbm$shrink, mejor_gbm$n_trees)
predecir_con[["Boosting (gbm)"]] <- function(nd)
  as.numeric(predict(m_gbm, newdata = nd, n.trees = mejor_gbm$n_trees))

rbind(boosting_validacion = metricas(valid$consumo, predecir_con[["Boosting (gbm)"]](valid)))



# PARTE 14: COMBINACIÓN DE MODELOS (AVERAGING Y STACKING)

set.seed(semilla)

modelos_base <- c("OLS con clima", "Splines / GAM", "Random Forest", "Boosting (gbm)")
oof_base <- cbind(oof_ols, oof_gam, oof_rf, oof_gbm)
colnames(oof_base) <- modelos_base
stopifnot(nrow(oof_base) == nrow(train), !anyNA(oof_base))

round(cor(train$consumo - oof_base), 2)

ajuste_nnls <- nnls(oof_base, train$consumo)
pesos <- coef(ajuste_nnls)
names(pesos) <- modelos_base
pesos <- pesos / sum(pesos)
round(pesos, 3)

predecir_con[["Averaging (promedio simple)"]] <- function(nd)
  rowMeans(sapply(modelos_base, function(m) predecir_con[[m]](nd)))
predecir_con[["Stacking (pesos OOF)"]] <- function(nd)
  as.numeric(sapply(modelos_base, function(m) predecir_con[[m]](nd)) %*% pesos)

round(c(apply(oof_base, 2, rmse_cv),
        "Averaging" = rmse_cv(rowMeans(oof_base)),
        "Stacking"  = rmse_cv(as.numeric(oof_base %*% pesos))), 5)
rbind(averaging_validacion = metricas(valid$consumo, predecir_con[["Averaging (promedio simple)"]](valid)),
      stacking_validacion  = metricas(valid$consumo, predecir_con[["Stacking (pesos OOF)"]](valid)))



# PARTE 15: TABLA OBLIGATORIA (VALIDACIÓN), INCERTIDUMBRE Y ELECCIÓN

set.seed(semilla)

pred_valid <- lapply(predecir_con, function(f) f(valid))
nombre_base <- "Baseline (promedio por zona)"

familia <- c("Baseline (promedio por zona)"  = "-",
             "Referencia cubica (velocidad)" = "Referencia",
             "OLS con clima"                 = "Lineal",
             "Ridge"                         = "Regularizado",
             "Lasso"                         = "Regularizado",
             "Elastic-net"                   = "Regularizado",
             "Splines / GAM"                 = "Flexible",
             "Arbol de regresion"            = "Arbol",
             "Random Forest"                 = "Ensemble",
             "Boosting (gbm)"                = "Ensemble",
             "Averaging (promedio simple)"   = "Ensemble",
             "Stacking (pesos OOF)"          = "Ensemble")

complejidad <- c("Baseline (promedio por zona)" = 1, "Referencia cubica (velocidad)" = 2,
                 "OLS con clima" = 3, "Ridge" = 3, "Lasso" = 3, "Elastic-net" = 3,
                 "Splines / GAM" = 4, "Arbol de regresion" = 4,
                 "Random Forest" = 5, "Boosting (gbm)" = 5,
                 "Averaging (promedio simple)" = 6, "Stacking (pesos OOF)" = 6)

stopifnot(setequal(names(pred_valid), names(familia)),
          setequal(names(pred_valid), names(complejidad)))

met <- t(sapply(pred_valid, function(p) metricas(valid$consumo, p)))

en_sub   <- zv == subgrupo
rmse_sub <- sapply(pred_valid, function(p) rmse(valid$consumo[en_sub], p[en_sub]))
mejora_pct <- 100 * (1 - met[, "RMSE"] / met[nombre_base, "RMSE"])

tabla <- data.frame(
  Modelo        = names(pred_valid),
  Familia       = familia[names(pred_valid)],
  RMSE          = round(met[, "RMSE"], 4),
  MAE           = round(met[, "MAE"],  4),
  R2            = round(met[, "R2"],   3),
  RMSE_subgrupo = round(rmse_sub, 4),
  vs_baseline   = ifelse(names(pred_valid) == nombre_base, "n/a",
                         sprintf("%.1f %% menos RMSE", mejora_pct)),
  row.names = NULL
)
print(tabla)
c(subgrupo = subgrupo, filas_en_validacion = sum(en_sub))

# DOS RMSE: general (todo el viaje) y subgrupo (velocidad entre 1 y 6 m/s)
rmse_gen_sub <- function(real, pred, vel) {
  en_sub <- zona_vel(vel) == subgrupo
  c(RMSE_general   = rmse(real, pred),
    RMSE_subgrupo  = rmse(real[en_sub], pred[en_sub]),
    filas_subgrupo = sum(en_sub))
}
dos_rmse <- data.frame(Modelo            = names(pred_valid),
                       RMSE_general      = round(met[, "RMSE"], 4),
                       RMSE_subgrupo_1a6 = round(rmse_sub, 4),
                       row.names = NULL)
dos_rmse$subgrupo_vs_baseline <- round(
  dos_rmse$RMSE_subgrupo_1a6 / dos_rmse$RMSE_subgrupo_1a6[dos_rmse$Modelo == nombre_base], 2)
print(dos_rmse)

# INCERTIDUMBRE (a): métricas en cada bloque de tiempo de validación
bloque_val <- cut(seq_len(nrow(valid)), breaks = n_bloques_cv, labels = FALSE)
mae_bloques <- sapply(pred_valid, function(p) {
  sapply(1:n_bloques_cv, function(b) mean(abs(valid$consumo[bloque_val == b] - p[bloque_val == b])))
})
incert_bloques <- data.frame(
  Modelo        = colnames(mae_bloques),
  MAE_promedio  = round(colMeans(mae_bloques), 4),
  MAE_desv      = round(apply(mae_bloques, 2, sd), 4),
  bloques_que_gana_al_baseline =
    colSums(mae_bloques < mae_bloques[, nombre_base]),
  row.names = NULL
)
print(incert_bloques)

# INCERTIDUMBRE (b): bootstrap por bloques
mejora_mae_ic <- function(real, p_base, p_mod, largo = 60, B = 1000, semilla_boot = semilla) {
  set.seed(semilla_boot)   # cada bootstrap parte de la misma semilla: no depende de lo que se ejecutó antes
  g    <- ceiling(seq_along(real) / largo)
  s_b  <- tapply(abs(real - p_base), g, sum)
  s_m  <- tapply(abs(real - p_mod),  g, sum)
  n_b  <- tapply(real, g, length)
  difs <- replicate(B, {
    i <- sample(seq_along(s_b), replace = TRUE)
    (sum(s_b[i]) - sum(s_m[i])) / sum(n_b[i])
  })
  c(mejora_MAE = mean(abs(real - p_base)) - mean(abs(real - p_mod)),
    IC95_inf   = unname(quantile(difs, 0.025)),
    IC95_sup   = unname(quantile(difs, 0.975)))
}
ic_mejora <- t(sapply(pred_valid[names(pred_valid) != nombre_base], function(p)
  mejora_mae_ic(valid$consumo, pred_valid[[nombre_base]], p)))
print(round(ic_mejora, 4))

# INCERTIDUMBRE (c): cobertura del OLS
ip <- predict(m_clima, newdata = valid, interval = "prediction", level = nivel_intervalo)
cobertura_ols <- mean(valid$consumo >= ip[, "lwr"] & valid$consumo <= ip[, "upr"])
cobertura_ols

# ELECCIÓN DEL SISTEMA FINAL
mejor <- names(which.min(met[, "RMSE"]))
dif_vs_mejor <- t(sapply(names(pred_valid), function(m)
  mejora_mae_ic(valid$consumo, pred_valid[[m]], pred_valid[[mejor]])))
colnames(dif_vs_mejor) <- c("MAE_extra_vs_mejor", "IC95_inf", "IC95_sup")
empata <- dif_vs_mejor[, "IC95_inf"] <= 0
eleccion <- data.frame(Modelo = names(pred_valid), RMSE = round(met[, "RMSE"], 4),
                       complejidad = complejidad[names(pred_valid)],
                       round(dif_vs_mejor, 4), empata_con_mejor = empata, row.names = NULL)
eleccion <- eleccion[order(eleccion$complejidad, eleccion$RMSE), ]
print(eleccion)

candidatos   <- eleccion$Modelo[eleccion$empata_con_mejor]
sugerido     <- candidatos[1]
modelo_final <- if (is.na(modelo_final_manual)) sugerido else modelo_final_manual
stopifnot(modelo_final %in% names(predecir_con))
c(mejor_en_validacion = mejor, sugerido_por_regla = sugerido, modelo_final = modelo_final)
if (sugerido != modelo_final)
  warning("La regla automática sugiere '", sugerido, "' pero el modelo final fijado es '", modelo_final, "'.")

err_final_valid <- valid$consumo - pred_valid[[modelo_final]]
ancho_zona <- tapply(abs(err_final_valid), zv, quantile, probs = nivel_intervalo)
round(ancho_zona, 4)



# PARTE 16: INTERPRETACIÓN (IMPORTANCIA, PDP E ICE) - SIN CAUSALIDAD

set.seed(semilla)
imp <- sort(importance(m_rf), decreasing = TRUE)
round(imp / max(imp), 3)
par(mar = c(4, 8, 2, 1))
barplot(rev(imp / max(imp)), horiz = TRUE, las = 1, col = "grey60",
        xlab = "Importancia relativa (permutación)", main = "Random Forest")
par(mar = c(5, 4, 4, 2) + 0.1)

f_final <- predecir_con[[modelo_final]]
pdp_ice <- function(variable, n_filas = 200, n_puntos = 30) {
  set.seed(semilla)   # misma muestra de filas en cada ejecución
  filas <- train[sample(nrow(train), n_filas), ]
  grilla <- seq(quantile(train[[variable]], 0.01), quantile(train[[variable]], 0.99),
                length.out = n_puntos)
  ice <- sapply(grilla, function(g) { nd <- filas; nd[[variable]] <- g; f_final(nd) })
  matplot(grilla, t(ice), type = "l", lty = 1, col = adjustcolor("grey40", 0.15),
          xlab = variable, ylab = "Consumo predicho (kg/s)",
          main = paste0("PDP/ICE: ", variable))
  lines(grilla, colMeans(ice), col = "red", lwd = 3)
  invisible(data.frame(valor = grilla, pdp = colMeans(ice)))
}
vars_pdp <- c("velocidad", "viento_vel", "viento_frente", "ola_alt")
par(mfrow = c(2, 2))
pdp_tablas <- lapply(vars_pdp, pdp_ice)
par(mfrow = c(1, 1))
names(pdp_tablas) <- vars_pdp



# PARTE 17: EVALUACIÓN FINAL EN TEST (UNA SOLA VEZ, AL FINAL)
# Se ejecuta solo si EJECUTAR_TEST = TRUE (PARTE 0). Nada anterior usó el test.
# Modelo final: el de la PARTE 15 / PARTE 0 (ajustado SOLO con entrenamiento).

if (EJECUTAR_TEST) {
  set.seed(semilla)
  p_test <- predecir_con[[modelo_final]](test)
  b_test <- predecir_con[[nombre_base]](test)
  zt     <- zona_vel(test$velocidad)

  # Métricas: validación contra test
  resultado_test <- rbind(
    "Baseline - validacion"      = metricas(valid$consumo, pred_valid[[nombre_base]]),
    "Baseline - test"            = metricas(test$consumo,  b_test),
    "Sistema final - validacion" = metricas(valid$consumo, pred_valid[[modelo_final]]),
    "Sistema final - test"       = metricas(test$consumo,  p_test))
  print(paste("Sistema final:", modelo_final))
  print(round(resultado_test, 4))

  # Los dos RMSE: general (todo el viaje) y subgrupo (1 a 6 m/s)
  dos_rmse_test <- rbind(
    "Baseline - validacion"      = rmse_gen_sub(valid$consumo, pred_valid[[nombre_base]],  valid$velocidad),
    "Baseline - test"            = rmse_gen_sub(test$consumo,  b_test,                     test$velocidad),
    "Sistema final - validacion" = rmse_gen_sub(valid$consumo, pred_valid[[modelo_final]], valid$velocidad),
    "Sistema final - test"       = rmse_gen_sub(test$consumo,  p_test,                     test$velocidad))
  print(round(dos_rmse_test, 4))

  mejora_rmse <- function(conjunto) {
    b <- dos_rmse_test[paste("Baseline -", conjunto),      c("RMSE_general", "RMSE_subgrupo")]
    f <- dos_rmse_test[paste("Sistema final -", conjunto), c("RMSE_general", "RMSE_subgrupo")]
    100 * (1 - f / b)
  }
  print(round(rbind(validacion = mejora_rmse("validacion"),
                    test       = mejora_rmse("test")), 1))   # % menos RMSE vs baseline

  # Mejora del MAE frente al baseline, con intervalo de confianza
  mejora_test <- mejora_mae_ic(test$consumo, b_test, p_test)
  print(round(mejora_test, 4))

  # Error (RMSE) por zona de velocidad
  rmse_zona_test <- rbind(
    baseline      = sapply(levels(zt), function(z) rmse(test$consumo[zt == z], b_test[zt == z])),
    sistema_final = sapply(levels(zt), function(z) rmse(test$consumo[zt == z], p_test[zt == z])),
    filas         = as.numeric(table(zt)))
  print(round(rmse_zona_test, 4))

  # Intervalo de predicción (ancho calibrado en VALIDACIÓN)
  ancho_t <- as.numeric(ancho_zona[as.character(zt)])
  dentro  <- test$consumo >= p_test - ancho_t & test$consumo <= p_test + ancho_t
  print(c(cobertura_total = mean(dentro), tapply(dentro, zt, mean)))

  # Gráfico: real vs predicho (si fuera perfecto, caería sobre la línea roja)
  plot(p_test, test$consumo, pch = 16, cex = 0.3,
       xlab = "Consumo predicho (kg/s)", ylab = "Consumo real (kg/s)",
       main = paste("Test:", modelo_final))
  abline(0, 1, col = "red", lwd = 2)
} else {
  message("PARTE 17 omitida: EJECUTAR_TEST = FALSE (el test sigue intacto).")
}



# PARTE 18: GUARDAR RESULTADOS Y CONFIGURACIÓN (REGISTRO)

dir.create(carpeta_salida, showWarnings = FALSE)
guardar <- function(objeto, nombre) write.csv(objeto, file.path(carpeta_salida, nombre), row.names = FALSE)
guardar(tabla,          "tabla_validacion.csv")
guardar(incert_bloques, "incertidumbre_bloques.csv")
guardar(data.frame(Modelo = rownames(ic_mejora), round(ic_mejora, 5)), "ic_mejora_vs_baseline.csv")
guardar(eleccion,       "eleccion_modelo_final.csv")
guardar(dos_rmse,        "dos_rmse_validacion.csv")
guardar(tab_relevancia, "coeficientes_relevancia.csv")
guardar(res_gam$grilla,   "grilla_cv_gam.csv")
guardar(res_arbol$grilla, "grilla_cv_arbol.csv")
guardar(res_rf$grilla,    "grilla_cv_random_forest.csv")
guardar(res_gbm,          "grilla_cv_boosting.csv")
guardar(data.frame(modelo = names(pesos), peso = round(pesos, 4)), "pesos_stacking.csv")
guardar(data.frame(variable = names(imp), importancia = imp), "importancia_rf.csv")

if (EJECUTAR_TEST) {
  guardar(data.frame(conjunto = rownames(resultado_test), round(resultado_test, 5)), "resultado_test.csv")
  guardar(data.frame(conjunto = rownames(dos_rmse_test), round(dos_rmse_test, 5)), "dos_rmse_test.csv")
}

writeLines(c(paste("semilla:", semilla),
             paste("prop_train:", prop_train, "| prop_valid:", prop_valid, "| hueco:", hueco),
             paste("filas train/valid/test:", nrow(train), nrow(valid), nrow(test)),
             paste("subgrupo:", subgrupo, "| umbral_relevancia:", umbral_relevancia),
             paste("GAM k_vel/k_otras:", res_gam$mejor$k_vel, "/", res_gam$mejor$k_otras),
             paste("Arbol cp/maxdepth:", res_arbol$mejor$cp, "/", res_arbol$mejor$maxdepth),
             paste("RF mtry/min.node.size/arboles:", res_rf$mejor$mtry, "/",
                   res_rf$mejor$min.node.size, "/", n_arboles_rf),
             paste("Boosting depth/shrinkage/arboles:", mejor_gbm$depth, "/",
                   mejor_gbm$shrink, "/", mejor_gbm$n_trees),
             paste("modelo final:", modelo_final, "| test ejecutado:", EJECUTAR_TEST),
             "", capture.output(sessionInfo())),
           file.path(carpeta_salida, "configuracion_y_sesion.txt"))

# HUELLA DE REPRODUCIBILIDAD: si dos ejecuciones muestran el mismo código md5, los resultados son idénticos
huella_obj <- list(validacion     = round(met, 6),
                   cv_gam         = round(res_gam$grilla$RMSE_cv, 6),
                   cv_arbol       = round(res_arbol$grilla$RMSE_cv, 6),
                   cv_rf          = round(res_rf$grilla$RMSE_cv, 6),
                   cv_boosting    = round(res_gbm$RMSE_cv, 6),
                   pesos_stacking = round(pesos, 6),
                   ic_mejora      = round(ic_mejora, 6),
                   importancia_rf = round(imp, 6),
                   test           = if (EJECUTAR_TEST) round(resultado_test, 6) else NULL)
archivo_huella <- file.path(carpeta_salida, "huella_reproducibilidad.txt")
writeLines(capture.output(print(huella_obj, digits = 10)), archivo_huella)
message("HUELLA DE REPRODUCIBILIDAD (md5): ", unname(tools::md5sum(archivo_huella)))

message("Ejecución finalizada con éxito. Resultados guardados en la carpeta '", carpeta_salida, "'.")