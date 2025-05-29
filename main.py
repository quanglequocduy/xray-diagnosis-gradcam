from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import uuid
import os

from gradcam_utils import generate_gradcam_and_upload

app = FastAPI()

@app.post("/gradcam")
async def gradcam_endpoint(
    image: UploadFile = File(...),
    model_name: str = Form(...)
):
    try:
        # Lưu ảnh tạm
        temp_filename = f"temp_{uuid.uuid4().hex}.jpeg"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Sinh GradCAM và upload Cloudinary
        image_url = generate_gradcam_and_upload(temp_filename, model_name)

        os.remove(temp_filename)

        return JSONResponse(status_code=200, content={"success": True, "gradcam_url": image_url})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
