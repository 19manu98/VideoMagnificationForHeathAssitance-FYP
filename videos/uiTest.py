import PySimpleGUI as sg

layout = [[sg.Text("PulseM: Heart Rate measuring using different approaches")],[sg.Button("PhotoPlethysmography")]]
layout2 = [[sg.Text("PulseM: Heart Rate measuring using different approaches")]]
window = sg.Window("PulseM",layout,finalize=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        window.close()
        break
    elif event == "PhotoPlethysmography":
        window.close()
        window2 = sg.Window("newWindow", layout2, finalize=True)
        while True:
            event, values = window2.read()
            if event == sg.WIN_CLOSED:
                window2.close()
                break