from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*",  # For testing purposes; replace with your frontend URL in production
    # e.g., "https://your-username.github.io",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Update with your frontend domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
