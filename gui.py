import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

class Gui():
	def __init__(self,master):
		# vars & gui elements
		whole_frame = tk.Frame(master)
		self.input_canvas = tk.Canvas(whole_frame,width=300,height=300)
		self.bg_img = ImageTk.PhotoImage(Image.open('bones.png'))
		self.bg_canvas_img = self.input_canvas.create_image((0,0),image=self.bg_img,anchor=tk.NW)
		
		self.current_x = tk.IntVar()
		self.current_y = tk.IntVar()

		info_frame = tk.Frame(master)
		self.x_label = tk.Label(info_frame,text="")
		self.y_label = tk.Label(info_frame,text="")
		# canvas elements
		self.vert_line = self.input_canvas.create_line(150,0,150,300,fill='white')
		self.horz_line = self.input_canvas.create_line(0,150,300,150,fill='white')
		# bindings

		self.input_canvas.bind("<Button-1>",self.get_coords)
		self.input_canvas.bind("<B1-Motion>",self.get_coords)

		# layout
		self.input_canvas.pack()
		self.x_label.pack(side=tk.LEFT)
		self.y_label.pack(side=tk.LEFT)
		info_frame.pack()
		whole_frame.pack()
		master.mainloop()
	def get_coords(self,event):
		# self.current_x.set(event.x)
		# self.current_y.set(event.y)
		self.update_display(event.x,event.y)
	def update_display(self,x,y):
		self.x_label.configure(text=x)
		self.y_label.configure(text=y)
		self.input_canvas.coords(self.vert_line,(x,0,x,300))
		self.input_canvas.coords(self.horz_line,(0,y,300,y))


if __name__=='__main__':
	root = tk.Tk()
	root.wm_resizable(0,0)
	root.title("sample_square")
	test = Gui(root)