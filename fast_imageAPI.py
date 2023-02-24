from fastapi import FastAPI, File, UploadFile

from src.infer_image import predict_image

app = FastAPI()

@app.post("/scorefile/")
def create_upload_file(file: UploadFile = File(...)):
    return predict_image(file.file)