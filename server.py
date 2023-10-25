from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

import mlflow
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

model = {
    'clf': None,
    'version': None
}


def load_model(version: Optional[str] = None):
    try:
        if version is None:
            with open('./last-run-id.txt') as f:
                version = f.read()
        clf = mlflow.sklearn.load_model(
            f'runs:/{version}/RandomForestModel'
        )
        model['version'] = version
        model['clf'] = clf
    except Exception as e:
        logging.error(e)


class MachineCondition(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    total_wear: float


class PredictionResult(BaseModel):
    model_version: str
    machine_condition: MachineCondition
    result: str


class RedeployRequest(BaseModel):
    version: str


class RedeployResponse(BaseModel):
    version: str


app = FastAPI()
load_model()


@app.post('/predict/')
async def create_item(machine_condition: MachineCondition) -> PredictionResult:
    type_map = {'H': 2, 'M': 1, 'L': 0}
    y = model['clf'].predict([[
        type_map[machine_condition.machine_type],
        machine_condition.air_temperature,
        machine_condition.process_temperature,
        machine_condition.rotational_speed,
        machine_condition.torque,
        machine_condition.total_wear
    ]])
    return PredictionResult(
        model_version=model['version'],
        machine_condition=machine_condition,
        result=y[0]
    )


@app.post('/redeploy')
async def redeploy(req: RedeployRequest) -> RedeployResponse:
    load_model(req.version)
    return RedeployResponse(
        version=model['version']
    )
