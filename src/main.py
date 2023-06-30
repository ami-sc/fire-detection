from threading import Thread
import motion2
import PySimpleGUI as sg
import cv2

layout = [[sg.Button('START', size=(10, 1), font='Helvetica 14')],
          [sg.Button('STOP', size=(10, 1), font='Helvetica 14')]]

window = sg.Window('Demo Application - OpenCV Integration',
                    layout, location=(800, 400))

i = 0
while True:

    event, values = window.read(timeout=20)
    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    elif event == 'START':
        motiondetection = Thread(target = motion2.fit, daemon = True)
        motiondetection.start()

    elif event == 'STOP':
        stopmotiondetection = Thread(target = motion2.stop)
        stopmotiondetection.start()
    
    i+=1
    print(i, event)
        


