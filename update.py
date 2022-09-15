from PIL import Image, ImageTk
import tkinter as tk
import cv2
from keras.models import model_from_json
import operator
#import hunspell
from string import ascii_uppercase
import tensorflow
import numpy as np



model=tensorflow.keras.models.load_model('model-bw.h5')




  self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language to Text Converter")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")
        self.panel = tk.Label(self.root)
        self.panel.place(x = 135, y = 50, width = 640, height = 400)
        self.panel2 = tk.Label(self.root) # initialize image panel
        self.panel2.place(x = 485, y = 50, width = 280, height = 280)
        
        self.T = tk.Label(self.root)
        self.T.place(x=280,y = 12)
        self.T.config(text = "Sign Language to Text",font=("courier",20,"bold"))
        self.panel3 = tk.Label(self.root) # Current SYmbol
        self.panel3.place(x = 300,y=460)
        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10,y = 460)
        self.T1.config(text="Character :",font=("Courier",20,"bold"))
        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 250,y=490)
        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 490)
        self.T2.config(text ="Word :",font=("Courier",20,"bold"))
        self.panel5 = tk.Label(self.root) # Sentence
        self.panel5.place(x = 270,y=520)
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10,y = 520)
        self.T3.config(text ="Sentence :",font=("Courier",20,"bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x = 250,y = 550)
        self.T4.config(text = "Suggestions",fg="red",font = ("Courier",20,"bold"))

        self.btcall = tk.Button(self.root,command = self.action_call,height = 0,width = 0)
        self.btcall.config(text = "About",font = ("Courier",14))
        self.btcall.place(x = 825, y = 0)

        self.bt1=tk.Button(self.root, command=self.action1,height = 0,width = 0)
        self.bt1.place(x = 26,y=590)
        # self.bt1.grid(padx = 10, pady = 10)
        self.bt2=tk.Button(self.root, command=self.action2,height = 0,width = 0)
        self.bt2.place(x = 360,y=590)
        # self.panel3.place(x = 10,y=660)
        # self.bt2.grid(row = 4, column = 1, columnspan = 1, padx = 10, pady = 10, sticky = tk.NW)
        self.bt3=tk.Button(self.root, command=self.action3,height = 0,width = 0)
        self.bt3.place(x = 725,y=590)
        # self.bt3.grid(row = 4, column = 2, columnspan = 1, padx = 10, pady = 10, sticky = tk.NW)
        self.bt4=tk.Button(self.root, command=self.action4,height = 0,width = 0)
        self.bt4.place(x = 145,y=620)
        # self.bt4.grid(row = bt1, column = 0, columnspan = 1, padx = 10, pady = 10, sticky = tk.N)
        self.bt5=tk.Button(self.root, command=self.action5,height = 0,width = 0)
        self.bt5.place(x = 525,y=620)
        # self.bt5.grid(row = 5, column = 1, columnspan = 1, padx = 10, pady = 10, sticky = tk.N)
        self.str=""
        self.word=""
        self.current_symbol="Empty"
        self.photo="Empty"
        self.video_loop()








def predict(test_image):
    test_image = cv2.resize(test_image, (128,128))
    result = model.predict(test_image.reshape(1, 128, 128, 1))
    prediction={}
    prediction['blank'] = result[0][0]
    inde = 1
    for i in ascii_uppercase:
        prediction[i] = result[0][inde]
        inde += 1
    # LAYER 1
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    current_symbol = prediction[0][0]
    
    if(current_symbol == 'blank'):
        for i in ascii_uppercase:
            ct[i] = 0
    ct[current_symbol] += 1
    if(ct[current_symbol] > 60):
        for i in ascii_uppercase:
            if i == current_symbol:
                continue
            tmp = ct[current_symbol] - ct[i]
            if tmp < 0:
                tmp *= -1
            if tmp <= 20:
                ct['blank'] = 0
                for i in ascii_uppercase:
                    ct[i] = 0
                return
        ct['blank'] = 0
        for i in ascii_uppercase:
            ct[i] = 0
        if current_symbol == 'blank':
            if blank_flag == 0:
                blank_flag = 1
                if len(str) > 0:
                    str += " "
                str += word
                word = ""
        else:
            if(len(str) > 16):
                str = ""
            blank_flag = 0
            word += current_symbol


cap=cv2.VideoCapture(0)

while True:
    frame = cap.read()
    cv2image = cv2.flip(frame, 1)
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
    current_image = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=current_image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    cv2image = cv2image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    predict(res)
    current_image2 = Image.fromarray(res)
    imgtk = ImageTk.PhotoImage(image=current_image2)
    panel2.imgtk = imgtk
    panel2.config(image=imgtk)
    panel3.config(text=current_symbol,font=("Courier",20))
    panel4.config(text=word,font=("Courier",20))
    panel5.config(text=str,font=("Courier",20))      

    if(cv2.waitKey(1)==13):
        break         

cap.release()
cv2.destroyAllWindows()