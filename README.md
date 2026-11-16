# 🎬 Sistema Bayesiano de Recomendación de Películas

Proyecto académico desarrollado en la **Universidad Industrial de Santander (UIS)** para la construcción de un sistema de recomendación cinematográfico basado en inferencia bayesiana con **PyMC**. El modelo integra el conjunto de datos histórico **MovieLens 100K** con información local recolectada mediante una encuesta de preferencias cinematográficas.

---

## Estructura del Repositorio

```text
Bayesian_recommendation_system/        
├── README.md                                                     # Documentación principal del proyecto        
├── Sistema_bayesiano_de_recomendación_de_películas_Informe.pdf   # Informe teórico y metodológico        
│        
├── aplicacion/                                                   # Código fuente, inferencia y documentación técnica        
│   ├── Consultas_dataset.ipynb                                   # Análisis exploratorio (EDA) y validación de datos        
│   ├── implementacion_computacional.md                           # Anexo de arquitectura en PyMC e ingeniería        
│   └── Modelo_Factorización_Matricial.ipynb                      # Entrenamiento NUTS, diagnósticos ArviZ y Top-N        
│        
└── data/                                                         # Conjuntos de datos del proyecto        
    ├── Encuesta de Preferencias Cinematográficas(1-57)(Sheet1).csv # Muestra primaria local (UIS)        
    │        
    ├── processed/                                                # Datos limpios y unificados listos para el modelo        
    │   ├── items_clean.csv                                       # Catálogo depurado de películas        
    │   ├── ratings_clean.csv                                     # Interacciones consolidadas (MovieLens + Encuesta)        
    │   └── users_clean.csv                                       # Usuarios históricos y encuestados reindexados        
    │        
    └── raw/                                                      # Archivos originales de MovieLens 100K sin modificar        
        ├── u.data        
        ├── u.genre        
        ├── u.info        
        ├── u.item        
        ├── u.occupation        
        └── u.user
```

---

## Fuentes de Datos y Preprocesamiento

### 1. MovieLens 100K (`data/raw/`)

- **Contenido original:** 100,000 calificaciones (escala de 1 a 5 estrellas) emitidas por 943 usuarios sobre 1,682 películas.
- **Limpieza aplicada:** Asignación de encabezados estructurales, eliminación de columnas nulas o irrelevantes (`video_release_date`, `imdb_url`), exclusión del registro sin identificar (`movie_id = 267`, `unknown`) y extracción del año de estreno (`release_year`).

### 2. Encuesta Local de Preferencias Cinematográficas

- **Recolección:** Formulario aplicado mediante Microsoft Forms a participantes entre 15 y 75 años bajo consentimiento explícito de tratamiento de datos personales (Ley 1581 de 2012).
- **Diseño del catálogo:** Selección equilibrada de **20 películas** presentes en MovieLens 100K, elegidas por su notoriedad en público joven y balance entre géneros (*Animación/Infantil, Ciencia Ficción, Acción/Aventura, Comedia, Drama/Romance, Crimen/Thriller y Terror*).
- **Transformación:** Filtrado estricto de respuestas con autorización aceptada (`"De acuerdo"`), eliminación de metadatos administrativos, estandarización demográfica (`age`, `gender`) y conversión de formato ancho (*wide*) a largo (*long*) descartando películas no vistas (`NA`).

### 3. Tablas Unificadas (`data/processed/`)

- **`users_clean.csv`:** Registro consolidado de usuarios de MovieLens (`user_id` 1–943) y encuestados locales (`user_id` 944+), diferenciados mediante la columna `source` (`movielens` / `encuesta`).
- **`items_clean.csv`:** Catálogo limpio de 1,681 películas con indicadores binarios para 18 géneros cinematográficos y año de lanzamiento.
- **`ratings_clean.csv`:** Interacciones observadas en formato largo (`user_id`, `movie_id`, `rating`, `timestamp`, `source`).

---

## Flujo de Trabajo de los Notebooks

1. **`Consultas_dataset.ipynb`**
2. Carga y depuración de `u.user`, `u.item`, `u.data` y del archivo CSV de la encuesta.
3. Mapeo automático de los títulos del formulario hacia sus identificadores oficiales (`movie_id`) en MovieLens 100K.
4. Generación de **10 gráficos exploratorios** (distribución etaria, ocupaciones por género, composición del catálogo por género, evolución temporal de estrenos, tasa de visualización por película y comparativa de calificaciones promedio entre la encuesta local y MovieLens).
5. Exportación de los archivos limpios hacia `data/processed/`.
6. 


**`Sistema_bayesiano_de_recomendación_de_películas.ipynb`**

1. Indexación contigua de usuarios y películas (`0` a $N-1$) para parametrización en PyMC.
2. Especificación de distribuciones a priori (*priors*), factores latentes y verosimilitud para estimar preferencias individuales y de grupo.
3. Muestreo mediante NUTS y diagnóstico de convergencia de cadenas MCMC con **ArviZ**.


---

## Tecnologías Utilizadas

- **Lenguaje:** Python 3
- **Manipulación y Análisis de Datos:** `pandas`, `numpy`
- **Visualización:** `matplotlib`, `seaborn`
- **Modelado Bayesiano y Diagnóstico:** `pymc`, `arviz`
- **Entorno de Ejecución:** Google Colab / Jupyter Notebook

---

### Archivo de Trazas (NetCDF)

Este archivo (`trace_final_movielens.nc`, ~490 MB) almacena todas las muestras posteriores generadas durante la inferencia MCMC con el algoritmo NUTS. Contiene las estimaciones distribuidas de los parámetros latentes, sesgos e hiperparámetros del modelo, permitiendo realizar análisis de convergencia en ArviZ, cálculos de RMSE y recomendaciones personalizadas sin tener que reentrenar el sistema.

## Autoría

 
