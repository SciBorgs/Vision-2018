import cv2
from base_camera import BaseCamera
from DistAngleVision import Vision


class Camera(BaseCamera):
    video_source = 1

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        vision = Vision(camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT), 30)

        while True:
            # read current frame
            _, img = camera.read()

            vision.processImg(img)
            processed = vision.getSourceImg()

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', processed)[1].tobytes()
