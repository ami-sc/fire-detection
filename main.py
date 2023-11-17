import os

from PySimpleGUI import Window, WIN_CLOSED, Button
from threading import Thread

from MotionDetection import MotionDetection

layout = [[Button("START", size=(10, 1), font="Helvetica 14")],
          [Button("STOP", size=(10, 1), font="Helvetica 14")]]

window = Window("Demo Application - Fire Detection",
                layout, location=(800, 400))

motion_detect = MotionDetection(video_path="http://66.119.104.155/mjpg/video.mjpg")
md_thread = Thread(target=motion_detect.fit, daemon=True)

while True:

    event, values = window.read(timeout=20)
    if event == "Exit" or event == WIN_CLOSED or event == "STOP":
        motion_detect.run = False
        md_thread.join()
        break

    elif event == "START":
        md_thread.start()

os._exit(0)
