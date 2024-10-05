# Actividad 4 - MLOps

## Rodríguez Orozco Enrique Jair
## Monsalve Gallegos Sebastian

## Descripción
El objetivo de este proyecto es entrenar, desplegar y validar un modelo de clasificación utilizando un flujo de trabajo MLOps. Se entrena un modelo de **Random Forest** optimizado con **Optuna**, se despliega una API con **FastAPI** para realizar predicciones, y se realizan pruebas de hipótesis de Kolmogorov-Smirnov para validar las distribuciones de las predicciones frente a los datos de entrenamiento.

## Tecnologías Utilizadas
- Python 3.8 o superior
- Bibliotecas: FastAPI, Uvicorn, Optuna, Pandas, Numpy, Scikit-learn, Scipy, Requests

## Prerrequisitos
1. Tener instalado Python 3.8 o superior en tu máquina.

## Instrucciones de Instalación

## Pasos para Windows:

1. Clona el repositorio:
   ```bash
   git clone git@github.com:Kike14/Actividad4-MLOps.git
2. Crea un entorno virtual:
   ```bash
    python -m venv .venv

3. Activa el entorno virtual:
    ```bash
    .venv\Scripts\activate

4. Actualiza pip:

   ```bash
    pip install --upgrade pip
5. Instala las dependencias:
   ```bash
    pip install -r requirements.txt
6. Ejecuta el script principal:
   ```bash
    python main.py
   
## Steps for Mac:
1. Clone the repository:
   ```bash
   git clone git@github.com:Kike14/Actividad4-MLOps.git
2. Create a virtual environment:
   ```bash
   python3 -m venv venv

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate

4. Upgrade pip:
   ```bash
   pip install --upgrade pip

5. Install dependencies:
   ```bash
   pip install -r requirements.txt

6. Run the main script:
   ```bash
   python main.py

## Estructura del Proyecto

tu_proyecto/
│
├── data/
│   ├── credit_train.csv
│   └── credit_pred.csv
│
├── models/
│   ├── random_forest.pkl
│   └── scaler.pkl
│
├── app.py
├── demo.py
├── main.py
├── requirements.txt
├── train.py
└── README.md

Contribuciones
Las contribuciones son bienvenidas. Por favor, envía un pull request siguiendo las pautas de estilo y las mejores prácticas.

Licencia
Este proyecto está bajo la licencia MIT. Para más detalles, revisa el archivo LICENSE.