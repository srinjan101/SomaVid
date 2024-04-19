import cv2
import numpy as np
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Load Keras model
model = load_model("facialemotionmodel.h5")

# Load face cascade classifier
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        try:
            for (p, q, r, s) in faces:
                image = gray[q:q + s, p:p + r]
                cv2.rectangle(frm, (p, q), (p + r, q + s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frm, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                            (0, 0, 255))
        except cv2.error:
            pass

        return av.VideoFrame.from_ndarray(frm, format='bgr24')

webrtc_streamer(
    key="emotion_detection_example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)