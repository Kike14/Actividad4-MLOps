import subprocess
import time
import os

# Ruta al ejecutable de Python en el entorno virtual
python_exec = os.path.join(".venv", "Scripts", "python.exe")

# Paso 1: Ejecutar el script de entrenamiento (train.py)
print("Entrenando el modelo...")
subprocess.run([python_exec, "train.py"])

# Paso 2: Levantar la API (app.py)
print("Levantando la API...")
api_process = subprocess.Popen([python_exec, "-m", "uvicorn", "app:app", "--port", "1234", "--reload"])

# Esperar un tiempo para asegurarse de que la API esté corriendo antes de seguir
time.sleep(5)

# Paso 3: Ejecutar el script de predicciones (demo.py)
print("Ejecutando las predicciones y pruebas de KS...")
subprocess.run([python_exec, "demo.py"])

# Paso 4: Cerrar el proceso de la API cuando se terminen las predicciones
api_process.terminate()
