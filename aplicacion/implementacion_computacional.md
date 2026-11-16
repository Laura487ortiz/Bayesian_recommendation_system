# Anexo Técnico: Implementación Computacional y Construcción del Modelo

Este documento detalla la arquitectura del código implementado para el Sistema Bayesiano de Recomendación de Películas, justificando cómo se traduce la formulación matemática teórica de la Factorización Matricial Probabilística Jerárquica en el entorno de programación probabilística **PyMC**.

---

## 1. Preparación de Datos y Estructura Tensorial

El primer paso para la construcción del modelo es la correcta disposición de los datos para el motor de álgebra tensorial (`PyTensor`). PyMC no acepta identificadores de texto o índices discontinuos; requiere que las entidades estén mapeadas de forma continua desde $0$ hasta $N-1$.

* **Indexación Contigua:** Se utilizó `pd.factorize()` para crear índices numéricos secuenciales tanto para los usuarios (`user_idx`) como para las películas (`movie_idx`).
* **Hold-out (80/20):** Se aplicó una máscara aleatoria para dividir el conjunto de datos en `train` (para entrenar el modelo) y `test` (para evaluar la capacidad de generalización y calcular el RMSE).
* **Coordenadas (`coords`):** Se definió un diccionario que asocia los índices numéricos con los identificadores originales de MovieLens, permitiendo que las matrices latentes tengan dimensiones semánticas (`dims`), lo cual es fundamental para la interpretabilidad de los resultados en **ArviZ**.

```python
coords = {          
    "user":   user_ids,      # IDs originales de usuarios          
    "movie":  movie_ids,     # IDs originales de películas          
    "factor": [f"f_{i}" for i in range(K)], # K=5 factores latentes          
}
```

---

## 2. Construcción del Grafo Acíclico Dirigido (DAG)

La celda del modelo construye simbólicamente la estructura de dependencias probabilísticas. Matemáticamente, cada calificación se estima como:

$$\hat{R}_{ij} = \mu + b_i + c_j + \mathbf{u}_i^T \mathbf{v}_j$$

La construcción en PyMC se divide en cuatro niveles jerárquicos:

### 2.1 Nivel Global (Intercepto)
Se define el promedio poblacional esperado de las calificaciones. Dado que la escala es de 1 a 5, el *prior* se centra en 3.

```python
mu = pm.Normal("mu", mu=3.0, sigma=1.0)
```

### 2.2 Nivel de Sesgos (Efectos Principales)
Se incorporan los sesgos individuales para capturar si un usuario es sistemáticamente severo o generoso, y si una película es universalmente aclamada o rechazada. Se utilizan distribuciones `HalfNormal` para las desviaciones estándar (escalas), ya que la varianza no puede ser negativa.

```python
sigma_b  = pm.HalfNormal("sigma_b", sigma=1.0) # Escala de severidad de usuarios          
sigma_c  = pm.HalfNormal("sigma_c", sigma=1.0) # Escala de calidad de películas
```

### 2.3 Nivel Latente (Interacciones Ocultas)
Para capturar afinidades específicas (ej. *"este usuario ama la ciencia ficción"*), se crean matrices de $K=5$ dimensiones para usuarios y películas. Estas se generan mediante distribuciones normales multivariadas escaladas.

```python
sigma_uv = pm.HalfNormal("sigma_uv", sigma=0.5) # Escala compartida para el espacio latente          

# Matrices latentes (Nota: Se usa ZeroSumNormal por razones de identificabilidad)          
b_raw = pm.ZeroSumNormal("b_raw", sigma=1.0, dims="user")          
c_raw = pm.ZeroSumNormal("c_raw", sigma=1.0, dims="movie")          
u_raw = pm.ZeroSumNormal("u_raw", sigma=1.0, dims=("factor", "user"))          
v_raw = pm.ZeroSumNormal("v_raw", sigma=1.0, dims=("factor", "movie"))          

# Aplicación de la escala          
u = u_raw * sigma_uv          
v = v_raw * sigma_uv
```

### 2.4 Nivel de Verosimilitud y Álgebra Vectorizada
En lugar de usar bucles `for` en Python (que serían extremadamente lentos para 100,000 registros), se utiliza **indexación avanzada de tensores**. Esto extrae los vectores correspondientes a cada interacción observada y calcula el producto punto simultáneamente para todo el conjunto de datos.

```python
# Producto punto vectorizado (u_i^T * v_j)          
interaccion = (u[:, u_idx] * v[:, m_idx]).sum(axis=0)          
          
# Ecuación completa          
rating_est  = mu + b_raw[u_idx] * sigma_b + c_raw[m_idx] * sigma_c + interaccion          
          
# Nodo de verosimilitud          
sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)          
pm.Normal("rating_obs", mu=rating_est, sigma=sigma_y, observed=y_obs)
```

---

## 3. Inferencia MCMC y Evaluación

El modelo se compila y se muestrea utilizando el algoritmo **NUTS (No-U-Turn Sampler)**. Para garantizar la convergencia, se configuraron 4 cadenas independientes con 1,000 iteraciones de ajuste (*tuning*) y 1,000 iteraciones de muestreo efectivo.

```python
trace = pm.sample(          
    draws=1000, tune=1000, chains=4,          
    target_accept=0.95,          
    nuts_sampler="numpyro", # Detallado en el anexo de ingeniería          
    nuts_sampler_kwargs={"chain_method": "vectorized"},          
)
```

### 3.1 Evaluación Predictiva (Hold-out Test)
Para calcular el RMSE real, se extraen las muestras posteriores y se predice la calificación de las interacciones en el conjunto `test`. Es fundamental destacar que **las predicciones se promedian "draw por draw"** antes de calcular el error; esto evita los problemas matemáticos de promediar directamente los vectores latentes (los cuales pueden presentar simetrías de rotación).

```python
def predecir(df, max_draws=500, lote=50):          
    # ... extracción de parámetros ...          
    for i in range(0, len(sel), lote):          
        s = sel[i:i + lote]          
        inter = np.einsum("snk,snk->sn", u_raw_s[s][:, ui, :], v_raw_s[s][:, mi, :])          
        pred = (mu_s[s, None] + b_raw_s[s][:, ui] * sb_s[s, None] + ... )          
        acum += pred.sum(axis=0)          
    return np.clip(acum / len(sel), 1, 5) # Forzado al rango [1, 5]
```

* **Resultado obtenido:** El modelo logró un RMSE de **0.9064**, frente a **1.1284** del *baseline* (media global). Esto representa una **mejora del 19.7%** en la precisión de las predicciones.

### 3.2 Generación de Recomendaciones Top-$N$
Para el despliegue del sistema, se utilizan las medias posteriores de los parámetros estimados para predecir la calificación de **todas las películas no vistas** por un usuario específico. Luego se ordenan y se extraen las mejores $N$ recomendaciones.

```python
# Predicción de scores para todo el catálogo          
scores = mu_hat + b_hat[user_idx] + c_hat + np.dot(u_hat[user_idx], v_hat.T)          

# Ocultar películas ya vistas en Train          
vistas = train[train['user_idx'] == user_idx]['movie_idx'].values          
scores[vistas] = -np.inf
```

---

## 4. Anexo de Ingeniería: Decisiones de Diseño y Optimización

Para que la formulación teórica anterior pudiera ejecutarse exitosamente con 100,000 registros y en un tiempo razonable, se tomaron las siguientes decisiones críticas de ingeniería que no están explícitas en la literatura matemática clásica:

### A. Uso de `ZeroSumNormal` para Romper Crestas de Identificabilidad
En la teoría, los parámetros se distribuyen como $\mathcal{N}(0, \sigma^2)$. Sin embargo, en la práctica, el intercepto global ($\mu$), los sesgos ($b_i$) y la parte constante de los factores latentes compiten por explicar la media de los datos. Esto genera "crestas" (*ridges*) en la distribución posterior que impiden que el *sampler* converja ($\hat{R} > 1.5$).
* **Decisión:** Usar `pm.ZeroSumNormal`. Esta distribución obliga matemáticamente a que la suma de los parámetros sea exactamente cero (ej. $\sum b_i = 0$), forzando al intercepto ($\mu$) a ser el único que absorbe la media global. Esto limpió los diagnósticos, llevando el $\hat{R}$ a un valor óptimo de `1.00`.

### B. Inversión de Dimensiones `(factor, user)` para JAX
La teoría sugiere matrices de tamaño `(usuario, factor)`. Sin embargo, el *backend* de JAX (usado para GPU) es mucho más eficiente en memoria si la dimensión de iteración rápida (el factor latente, $K=5$) está al principio.
* **Decisión:** Se definieron las dimensiones como `dims=("factor", "user")`. Esto permite que la indexación avanzada `u[:, u_idx]` sea óptima en los hilos de la GPU, acelerando drásticamente el cómputo.

### C. Migración a GPU con NumPyro/JAX
El *sampler* NUTS estándar de PyMC en CPU colapsó (*Step size 0.000*) al intentar explorar un espacio de $\sim 13,000$ parámetros.
* **Decisión:** Se cambió el *backend* a `nuts_sampler="numpyro"` y se ejecutó en una GPU T4 de Google Colab utilizando `chain_method="vectorized"`. Esto redujo el tiempo de muestreo de horas a solo **13.7 minutos**, sin alterar la estructura matemática del modelo NUTS.

### D. Escala Latente Compartida (`sigma_uv`)
Tener varianzas separadas para usuarios ($\sigma_u$) y películas ($\sigma_v$) genera una invarianza de escala en el producto punto ($\sigma_u \cdot \sigma_v$), creando otra cresta problemática.
* **Decisión:** Se utilizó una sola hiperescala `sigma_uv` para ambos. El modelo no pierde capacidad representacional y gana una total identificabilidad en el espacio latente.