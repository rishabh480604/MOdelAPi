from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from model import detect_emotion

# Create FastAPI app
app = FastAPI()

# Enable CORS (allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test endpoint to check if the API is working
@app.get("/")
async def test():
    return {
        "message": "working fine",
        "status": "success"
    }

# Endpoint to receive and process an image
@app.post("/upload-image")
async def upload_image(image: UploadFile = File(...)):
    try:
        # Ensure the file is an image
        img = Image.open(io.BytesIO(await image.read()))
        
        # Call the detect_emotion function to analyze the image
        emotion = detect_emotion(img)

        return {
            "message": "Image processed successfully",
            "emotion": emotion
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


# Run the app if you're running locally using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
















"""from flask import Flask, request, jsonify
from PIL import Image
import io
from model import detect_emotion
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Endpoint to receive image and process it
@app.route('/')
def test():
    return jsonify({
        'message':"working fine",
        'status':'success'
    }

    )
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

        # Get the image from the request
    image = request.files['image']

    # Convert the image to a PIL format to work with it
    img = Image.open(io.BytesIO(image.read()))

    # Call the detect_emotion function to analyze the image
    emotion = detect_emotion(img)

    return jsonify({
        'message': 'Image processed successfully',
        'emotion': emotion
    })


if __name__ == '__main__':
    app.run(debug=True,port=8000)
"""