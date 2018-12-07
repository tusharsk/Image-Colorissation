
# coding: utf-8

# In[2]:


from tkinter import filedialog
from tkinter import *
from PIL import ImageTk,Image
from predict import predict
from threading import Thread

filename = ""
root = Tk()
root.title("IMAGE COLORIZATION")

def selectImage(canvas) :
    #button2.config(bg="white", state=DISABLED,)
    global filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                                       filetypes=(
                                                                           ("jpeg files", '*.jpeg'), ("all files", "*.*")))
    image = Image.open(filename)
    image = image.resize((224, 224), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    label = Label(image=img)
    label.image = img  # keep a reference!
    # label.pack()
    canvas.create_image(30, 50, anchor=NW, image=img)
    button2.config(bg="white", state=DISABLED, disabledforeground=canvas.cget('bg'))
    canvas.create_rectangle(310, 50, 534, 274, fill="white", outline="white")
    button1 = Button(root, text="COLORIZE", command=lambda: testing_function(canvas,button1))
    button1.configure(width=10, activebackground="#33B5E5", relief=FLAT)
    button1_window = canvas.create_window(240, 365, anchor=NW, window=button1)
    
def testing_function(canvas,button):
    button.config(bg='mistyrose4')
    thread1=Thread(target=colorizeImage,args=(canvas,))
    thread1.start()
    button2.config(bg='systembuttonface',state="normal")

def colorizeImage(canvas) :
    canvas.create_text(420, 20, fill="black", font="Times 12 italic bold",
                       text="The image on RGB scale is : ")
    global filename
    output_path=predict(filename)
    image = Image.open(output_path)
    image = image.resize((224,224), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(image)
    label = Label(image=img)
    label.image = img
    canvas.create_image(310, 50, anchor=NW, image=img)
    
canvas = Canvas(root, bg='white', width=560, height=400)
canvas.pack()
canvas.create_line(280, 0, 280, 350, fill="black", width=1)
canvas.create_line(0, 350, 560,350, fill="black", width=1)

canvas.create_text(140,20,fill="black",font="Times 12 italic bold",
                        text="Select the image to be colorized")

canvas.create_rectangle(30, 50, 253, 273, fill="white", dash=(3,5))

button1 = Button(root, text="SELECT IMAGE", command=lambda: selectImage(canvas))
button1.configure(width=10, activebackground="#33B5E5", relief=FLAT)
button1_window = canvas.create_window(105, 300, anchor=NW, window=button1)

button2 = Button(root, text="EXIT", command = root.destroy,bg="WHITE",state=DISABLED,disabledforeground=canvas.cget('bg'))
button2.configure(width=10, activebackground="#33B5E5", relief=FLAT)
button2_window = canvas.create_window(380, 300, anchor=NW, window=button2)

root.mainloop()

