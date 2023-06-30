from threading import Thread
import motion2
import PySimpleGUI as sg
import cv2

from motionclass import motiondetection

layout = [[sg.Button('START', size=(10, 1), font='Helvetica 14')],
          [sg.Button('STOP', size=(10, 1), font='Helvetica 14')]]

window = sg.Window('Demo Application - OpenCV Integration',
                    layout, location=(800, 400))

motion_detect = motiondetection()

motiondetection = Thread(target = motion_detect.fit, daemon = True)

i = 0
while True:

    event, values = window.read(timeout=20)
    if event == 'Exit' or event == sg.WIN_CLOSED or event == "STOP":
        motion_detect.fitting = False
        motiondetection.join()
        break

    elif event == 'START':
        motiondetection.start()

    i+=1
    print(i, event)



