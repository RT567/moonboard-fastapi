from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class HoldsRequest(BaseModel):
    holds: List[str]

class GradeResponse(BaseModel):
    grade: str

@app.post("/grade_climb", response_model=GradeResponse)
async def grade_climb(Request: HoldsRequest):
    return{"grade": "V5"}


@app.get("/")
async def root():
    return {"message": "Hello World"}
