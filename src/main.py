from motion2 import Motiondetection
from threading import Thread
import PySimpleGUI as sg
import cv2

t = Motiondetection()

layout = [[sg.Button('Show', size=(10, 1), font='Helvetica 14')],
          [sg.Button('Stop', size=(10, 1), font='Helvetica 14')],
          [sg.Button('uiShow', size=(10, 1), font='Helvetica 14')],
          [sg.Image(filename='', key='image')]]

window = sg.Window('Demo Application - OpenCV Integration',
                    layout, location=(800, 400))

i = 0
showmotion = False
while True:

    event, values = window.read(timeout=20)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    elif event == 'Show':
        motiondetection = Thread(target = t.fit)
        motiondetection.start()
    
    elif event == 'Stop':
        motiondetection._stop()

    elif event == 'uishow':
        showmotion=True
        motiondetection = Thread(target = t.fit, args=True)

    if showmotion:
        frame = t.frame
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
    

    

    

    
    i+=1
    print(i, event)
        


