import pickle
from fastapi import FastAPI
import uvicorn

input_file = "pipeline_v2.bin"
with open(input_file, 'rb') as f:
    dv,model = pickle.load(f)

app = FastAPI(title="Lead Score")

@app.post("/predict")
def predict(client: dict):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    return{"probability":float(y_pred)}

if __name__ == "__main__":
    uvicorn.run("predict:app", host='0.0.0.0', port=9696)