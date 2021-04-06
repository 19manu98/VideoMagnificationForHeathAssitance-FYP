from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
from Photoplethysmography import Photoplethysmography
from EVM import EVM

window = Tk()
window.geometry('400x470')
windowframe = Frame(window, background="black",relief=RIDGE, borderwidth=2)
windowframe.pack(fill=BOTH, expand=1)
window.title('PulseM')
label = Label(windowframe, text="PulseM: \nGet your Pulse",bg ="black", fg="green",font=('Courier 25 bold'))
label.pack(side=TOP)

def helpButton():
    tkinter.messagebox.showinfo("Usage","In order to use the Program please wait\n 20-25 seconds with your face in the middle of the camera.\n or choose a video")


def Contributors():
    tkinter.messagebox.showinfo("Contributors", "\n1.Iosif Balasca\nC17719309")


menu = Menu(window)
window.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="How to use", menu=subm1)
subm1.add_command(label="How to use", command=helpButton)

subm2 = Menu(menu)
menu.add_cascade(label="About", menu=subm2)
subm2.add_command(label="Contributors", command=Contributors)


def exitButton():
    exit()


def WebCamPhotoplethysmography():
    tkinter.messagebox.showinfo("Usage", "you need to stay still for around 20/25 seconds than the pulse will be shown to you\n press ESC to close the window")
    Photoplethysmography('webcam')


def VideoPhotoplethysmography():
    filepth = filedialog.askopenfilename()
    tkinter.messagebox.showinfo("Usage",
                                "the bpm will be displayed after 20/25 seconds \n press ESC to close the window")
    try:
        Photoplethysmography(filepth)
    except:
        tkinter.messagebox.showinfo("Error",
                                    "Sorry the file is not axcepted\nPlease try againg using a different file")

def WebCamEVM():
    tkinter.messagebox.showinfo("Usage", "you need to stay still for around 20/25 seconds than the pulse will be shown to you\n press ESC to close the window")
    EVM('webcam')


def VideoEVM():
    filepth = filedialog.askopenfilename()
    tkinter.messagebox.showinfo("Usage",
                                "the bpm will be displayed after 20/25 seconds \n press ESC to close the window")
    EVM(filepth)


but1 = Button(windowframe, padx=5, pady=5, width=45, bg='white', fg='black', relief=GROOVE, command=WebCamPhotoplethysmography, text='WebCam Photoplethysmography',
              font=('Courier 10 bold'))
but1.place(x=8, y=104)

but2 = Button(windowframe, padx=5, pady=5, width=45, bg='white', fg='black', relief=GROOVE, command=VideoPhotoplethysmography, text='Video Photoplethysmography',
              font=('Courier 10 bold'))
but2.place(x=8, y=144)

but3 = Button(windowframe, padx=5, pady=5, width=45, bg='white', fg='black', relief=GROOVE, command=WebCamEVM, text='WebCam EVM',
              font=('Courier 10 bold'))
but3.place(x=8, y=184)

but4 = Button(windowframe, padx=5, pady=5, width=45, bg='white', fg='black', relief=GROOVE, command=VideoEVM, text='Video EVM',
              font=('Courier 10 bold'))
but4.place(x=8, y=224)

window.mainloop()