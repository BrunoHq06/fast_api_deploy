from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import onnx
import onnxruntime as rt
import numpy as np

# Criação da app FastAPI
app = FastAPI()

# Carregando o modelo treinado
with open("./artifacts/stroke_rf.onnx", "rb") as f:
    onnx_model = onnx.load(f)

# Definindo a classe que representa o corpo da requisição para a rota /predict
class PredictRequest(BaseModel):
    age: float
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    Residence_type: str


def transform_input(input_data: dict):
    transformed_data = {}
    
    for key, value in input_data.items():
        if isinstance(value, (int, float)):  # Se for numérico, converte para array de float32
            transformed_data[key] = np.array([[value]], dtype=np.float32)
        else:  # Se for categórico, converte para array de object
            transformed_data[key] = np.array([[value]], dtype=object)
    
    return transformed_data


# Rota para verificar a saúde da API
@app.get("/api/health")
def health():
    return {"message": "Estou saudável"}

# Rota para fazer predições
@app.post("/api/predict")
def predict(request: PredictRequest):
    # Convertendo o corpo da requisição (JSON) para um dicionário
    request_data = request.dict()

    # Convertendo as features para um formato de array que o modelo aceita
    features = transform_input(request_data)

    # Realizando a predição com o modelo carregado
    sess = rt.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    prediction = sess.run(None, features)

    # Retornando o resultado da predição
    return {"prediction": int(prediction[0][0])}

