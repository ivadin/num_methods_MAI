import l2 as l
import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
LARGE_FONT = ("Verdana", 12)


class Application(tk.Tk):

    def __init__(self, *args, **kwargs):  # конструктор класса
        # позиционные и именные аргументы
        tk.Tk.__init__(self, *args, **kwargs)

        label = tk.Label(self, text="Лабораторная №2", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        im = Image.open('img.gif')
        render = ImageTk.PhotoImage(im)
        img = tk.Label(self, image=render)
        img.image = render
        img.pack(pady=10, padx=10)

        label0 = tk.Label(self, text='Введите параметр "t" > 0')
        label0.pack()
        self.ent0 = tk.Entry(self, width=20, bd=3)
        self.ent0.pack()

        label2 = tk.Label(self, text='Введите число интервалов по X')
        label2.pack()
        self.ent2 = tk.Entry(self, width=20, bd=3)
        self.ent2.pack()

        label3 = tk.Label(self, text='Введите число интервалов по T')
        label3.pack()
        self.ent3 = tk.Entry(self, width=20, bd=3)
        self.ent3.pack()

        label4 = tk.Label(self, text='Выберете метод')
        label4.pack()
        self.cb1 = ttk.Combobox(self,
                                values=[u"Явный",
                                        u"Неявный"],
                                height=3)
        self.cb1.set('Не выбрано')
        self.cb1.pack()

        label41 = tk.Label(
            self, text='Выберете порядок аппроксимацию'
            ' второго начального условия')
        label41.pack()
        self.cb11 = ttk.Combobox(self,
                                 values=[u"Первый",
                                         u"Второй"],
                                 height=3)
        self.cb11.set('Не выбрано')
        self.cb11.pack()

        label5 = tk.Label(
            self, text='Выберете аппроксимацию граничных условий')
        label5.pack()
        self.cb2 = ttk.Combobox(self,
                                values=[u"Двухточечный(первый порядок)",
                                        u"Трехточечный(второй порядок)",
                                        u"Двухточечный(второй порядок)"],
                                height=3)
        self.cb2.set('Не выбрано')
        self.cb2.pack()

        label6 = tk.Label(self, text='Введите момент времени от 0 до t')
        label6.pack()
        self.ent4 = tk.Entry(self, width=20, bd=3)
        self.ent4.pack()

        self.button = ttk.Button(
            self, text="Построить!",
            command=self.GetParams)

        self.button.pack(pady=10, padx=10)
        self.button.pack()

    def ErrorWindow(self, note):
        err_win = Tk()
        err_note = tk.Label(
            err_win, text=note,
            width=50, height=10)
        err_win.title('ERROR')
        err_note.pack(pady=10, padx=10)

    def GetParams(self):
        x0 = 0
        xl = 1

        param_t = float(self.ent0.get())
        m = int(self.ent2.get())
        n = int(self.ent3.get())
        method = self.cb1.get()
        apr_board = self.cb11.get()
        apr = self.cb2.get()
        ans_t = float(self.ent4.get())
        aprox = 0

        space_step = (xl - x0) / (m - 1)
        time_step = param_t / (n - 1)

        if param_t < 0:
            note = 'Ошибка!\nПараметр "t" должен быть больше 0!'
            self.ErrorWindow(note)
            return
        elif ans_t > param_t:
            note = 'Ошибка!\nПараметр момент времени должен быть меньше "t"!'
            self.ErrorWindow(note)
            return
        elif time_step**2 / space_step**2 >= 1\
                and method == 'Явный':
            note = 'Ошибка!\nПри таких мараметрах Явный метод не устойчив!\n\
            Пожалуйста, измените параметры сетки.'
            self.ErrorWindow(note)
            return
        if apr == 'Двухточечный(первый порядок)':
            aprox = 1
        elif apr == 'Трехточечный(второй порядок)':
            aprox = 2
        elif apr == 'Двухточечный(второй порядок)':
            aprox = 3
        if apr_board == 'Первый':
            aprox_board = 1
        elif apr_board == 'Второй':
            aprox_board = 2

        if method == 'Явный':
            l.explicit(param_t, m, n, aprox, aprox_board, ans_t)
        elif method == "Неявный":
            l.implicit(param_t, m, n, aprox, aprox_board, ans_t)

app = Application()
app.title('Лабораторная №2. (c) Демин И.А.')
app.mainloop()
