from re import search
from flask import Flask, escape, request,send_file
import VIPER
app = Flask(__name__)

@app.route('/', methods=['POST'])
def run():
    if request.method == "POST":
        imageFile = request.args.get('image', '')
        modifier = request.args.get("modifier","")
        relationship = request.args.get("relationship","")

        VIPER.command_line(imageFile, modifier, int(relationship))
        return send_file('image.jpg', mimetype='image/jpg')
    


"""
move the blurring to the device. make the device to the work of blurring the photos.
"""