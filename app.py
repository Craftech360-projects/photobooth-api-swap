from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import shutil
import cv2
import os
from insightface.app import FaceAnalysis
import insightface
import uuid
import gdown

# Initialize FastAPI app
app = FastAPI()

# Initialize FaceAnalysis and swapper model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Download 'inswapper_128.onnx' file using gdown if not already downloaded
model_url = 'https://drive.google.com/uc?id=1HvZ4MAtzlY74Dk4ASGIS9L6Rg5oZdqvu'
model_output_path = 'inswapper/inswapper_128.onnx'
if not os.path.exists(model_output_path):
    gdown.download(model_url, model_output_path, quiet=False)

swapper = insightface.model_zoo.get_model('inswapper/inswapper_128.onnx', download=False, download_zip=False)

# Directory setup for uploads and results
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def simple_face_swap(sourceImage, targetImage, face_app, swapper):
    # Perform face detection
    facesimg1 = face_app.get(sourceImage)
    facesimg2 = face_app.get(targetImage)
    
    if len(facesimg1) == 0 or len(facesimg2) == 0:
        return None  # No faces detected
    
    # Get first face detected in each image
    face1 = facesimg1[0]
    face2 = facesimg2[0]

    # Perform face swapping
    img1_swapped = swapper.get(sourceImage, face1, face2, paste_back=True)
    
    return img1_swapped

@app.post("/api/swap-face/")
async def swap_faces(sourceImage: UploadFile = File(...), targetImage: UploadFile = File(...)):
    # Save uploaded images to the upload folder
    img1_path = os.path.join(UPLOAD_FOLDER, sourceImage.filename)
    img2_path = os.path.join(UPLOAD_FOLDER, targetImage.filename)

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(sourceImage.file, buffer)
    with open(img2_path, "wb") as buffer:
        shutil.copyfileobj(targetImage.file, buffer)

    # Read images from saved paths
    sourceImage = cv2.imread(img1_path)
    targetImage = cv2.imread(img2_path)

    # Perform face swap
    swapped_image = simple_face_swap(sourceImage, targetImage, face_app, swapper)
    if swapped_image is None:
        raise HTTPException(status_code=500, detail="Face swap failed")

    # Save swapped image to results folder with unique filename
    result_filename = str(uuid.uuid4()) + '.jpg'
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, swapped_image)

    # Return the swapped image as response
    return FileResponse(result_path)

# Run the FastAPI app with Uvicorn server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
