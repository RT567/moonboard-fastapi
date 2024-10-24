import subprocess
import os
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*",  # For testing purposes; replace with your frontend URL in production
    # e.g., "https://your-username.github.io",
    "https://rt567.github.io/moonboard/"
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

# @app.post("/grade_climb", response_model=GradeResponse)
# async def grade_climb(Request: HoldsRequest):
#     return{"grade": "V5"}

@app.post("/grade_climb", response_model=GradeResponse)
async def grade_climb(request: HoldsRequest):
    # Build the command to call the other script
    holds = request.holds  # List of holds
    script_path = os.path.join(os.path.dirname(__file__), "python-image-creator", "grade-singular-climb-from-holdlist.py")
    command = ["python", script_path] + holds

    try:
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Process the output to extract the grade
        output = result.stdout
        grade_line = next((line for line in output.splitlines() if "Grade predicted as:" in line), None)
        if grade_line:
            grade = grade_line.split(":")[-1].strip()
            return {"grade": grade}
        else:
            return {"grade": "Could not determine grade from output."}
    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess
        return {"grade": f"Error occurred: {e.stderr}"}


@app.get("/")
async def root():
    return {"message": "Hello World"}
