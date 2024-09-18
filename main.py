from flask import Flask, request, jsonify
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
