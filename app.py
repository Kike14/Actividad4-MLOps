import pickle

from fastapi import FastAPI
import uvicorn
import pandas as pd

app = FastAPI()

model = pickle.load(open('./models/random_forest.pkl', 'rb'))

@app.get("/")
def greet(name: str):
    return {
        "message": f"Greetings, {name}"
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK"
    }

@app.post("/predict")
def predict(data: list[float]):
    X = [{
        f"X{i+1}": x
        for i, x in enumerate(data)
    }]
    df = pd.DataFrame.from_records(X)
    prediction = model.predict(df)
    return {
        "prediction": int(prediction[0])
    }


if __name__ == "__main__":
    uvicorn.run("app:app", port = 1234, reload = True)