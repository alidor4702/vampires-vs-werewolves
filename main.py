# main.py
import tkinter as tk
from gui.menu import MainMenu

def main():
    root = tk.Tk()
    MainMenu(root)
    root.mainloop()

if __name__ == "__main__":
    main()
