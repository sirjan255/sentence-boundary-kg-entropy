from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io

router = APIRouter()

@router.post("/check_embeddings/")
async def check_embeddings(embeddings_file: UploadFile = File(...)):
    """
    Accepts a .npy file of embeddings, returns shape and dtype for frontend display.
    """
    if not embeddings_file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Please upload a .npy file.")
    try:
        # Read file into memory and load numpy array
        contents = await embeddings_file.read()
        buf = io.BytesIO(contents)
        embeddings = np.load(buf, allow_pickle=True)
        shape = embeddings.shape
        dtype = str(embeddings.dtype)
        return JSONResponse({"shape": shape, "dtype": dtype})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read embeddings: {str(e)}")