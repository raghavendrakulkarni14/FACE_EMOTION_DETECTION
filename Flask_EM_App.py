from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model  
from tensorflow.keras.preprocessing import image  
  

#load model  
model = load_model('emotion detection model.h5')

classes = ['angry','fear','happy','neutral']
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  


app = Flask(__name__)

font = cv2.FONT_HERSHEY_TRIPLEX # Defining the font

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, img = camera.read()
        if not success:
            break
        else:
            grey = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces = facecascade.detectMultiScale(grey,1.3,1) # Region of interest of detected image
            
        
        for (x,y,w,h) in faces:
            x = x - 5
            y = y + 7
            w = w + 10
            h = h + 2
            roi_grey =  grey[y:y+h,x:x+w] # Cropping gray color image
            roi_color = img[y:y+h,x:x+w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Draw rectangle over the face
            faces = facecascade.detectMultiScale(roi_grey)
            if len(faces)==():
                raise IOError('Face not detected')
                
            else:
                for (ex,ey,ew,eh) in faces:
                    face_roi = roi_color[ey:ey+eh,ex:ex+ew]
                    
            final_image = cv2.resize(roi_color,(48,48)) # Image is resized to (48,48)
            final_image = np.expand_dims(final_image,axis=0) # array is expanded by inserting axis
            final_image = final_image/255.0 # Scaling of the image
        
            predictions=model.predict(final_image) # Making predictions
            label = classes[predictions.argmax()] # Finding label of the class which has highest probability
            cv2.putText(img,label, (50,60),font,2, (255,200,0),2)  
          
            
            ret, buffer = cv2.imencode('.jpg', img)
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)