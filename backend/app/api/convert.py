from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import json
import io

router = APIRouter()

@router.post("/convert_predicted_to_strings/")
async def convert_predicted_to_strings(predicted: UploadFile = File(...)):
    """
    Accepts a JSON file with predicted boundaries (dict: key -> list of dicts),
    returns a dict: key -> list of node strings.
    """
    try:
        if not predicted.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Please upload a .json file.")
        # Read and parse the uploaded file
        contents = await predicted.read()
        pred = json.loads(contents.decode("utf-8"))
        # Convert key -> list of dicts -> key -> list of node strings
        converted = {k: [d["node"] for d in v if isinstance(d, dict) and "node" in d] for k, v in pred.items()}
        return JSONResponse(content=converted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")