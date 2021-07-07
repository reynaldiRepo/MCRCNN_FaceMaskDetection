from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit
import cv2
from keras import models
import numpy as np
import base64
import json

# //import app
from Main import *

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app,always_connect=True)
C, model_rpn, model_classifier = setupmodel(type="MCRCNN", doCompile= False)


isProcess = False

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

@app.route('/')
def home():
   return render_template('./index.html')

@socketio.on('message')
def handle_message(data):
    print('received message:')

def log():
    print('output was received!')

@socketio.on('input')
def handle_input(data): 
   global isProcess
   if isProcess == True:
      return 0
   try:
      img = readb64(data['data'])
      print(img.shape)
      box = doPredict(C, model_rpn, model_classifier, img)
      print(box)
      emit('output', json.dumps(box), callback=log)
      isProcess = False
      del img;
      return 0
   except Exception as e:
       print(e)
       return 0

socketio.run(app)