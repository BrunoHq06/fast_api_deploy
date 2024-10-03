# API de Previsão de Infartos

Este projeto consiste em uma API REST que faz previsões sobre a probabilidade de um paciente sofrer um infarto, utilizando um modelo de Machine Learning treinado com Random Forest e exportado no formato ONNX.

## Pré-requisitos

- **Docker** instalado na máquina
- **Docker Compose** instalado

## Instruções para rodar o projeto

### 1. Clonar o repositório

Clone o repositório do projeto para sua máquina local:

Nesse repositório, temos todos os arquivos utilizados para construção da imagem, caso queira ter acesso a todos estes para que posso analizá-lo mais profundamente, fique à vontade para clonar o repositório inteiro

```bash
git clone https://github.com/BrunoHq06/fast_api_deploy.git
cd fast_api_deploy
```

### 2. Construir e rodar o container Docker

Use o Docker Compose para construir e rodar a aplicação. Este comando irá construir a imagem do Docker a partir do `Dockerfile`, instalar as dependências e iniciar a API:

```bash
docker-compose up --build
```

### 3. Testar a API

Após rodar o comando acima, a API estará disponível no seguinte endereço:

```
http://localhost:8080
```

#### Rota de saúde

Você pode verificar se a API está funcionando acessando a rota de saúde:

```
GET /api/health
```

Resposta esperada:

```json
{
  "message": "Estou saudável"
}
```

#### Rota de predição

Para fazer uma previsão, utilize a rota `/api/predict\`. A requisição deve ser feita via `POST` com o seguinte formato JSON:

```json
{
  "age": 45,
  "avg_glucose_level": 85.2,
  "bmi": 25.6,
  "smoking_status": "formerly smoked",
  "Residence_type": "Urban"
}
```

## Dependências

As principais dependências do projeto estão listadas no arquivo `requirements.txt` e incluem:

- `FastAPI`
- `Uvicorn`
- `onnxruntime`
- `pydantic`
- `numpy`

## Estrutura do projeto

```bash
.
├── artifacts
│   └── stroke_rf.onnx     # Modelo ONNX treinado
├── datasets
│   ├── sample.csv
│   ├── test.csv
│   └── train.csv
├── notebooks
│   └── model_training.ipynb  # Notebook com o treinamento do modelo
├── .dockerignore           # Arquivos a serem ignorados pelo Docker
├── api.py                  # Arquivo principal da API FastAPI
├── docker-compose.yaml     # Arquivo de configuração do Docker Compose
├── Dockerfile              # Dockerfile para criação da imagem
├── xpinc.tar               # Imagem criada
└── requirements.txt        # Dependências do projeto
```
