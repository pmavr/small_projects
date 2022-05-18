import tkinter
from tkinter import ttk
import enum


class ColumnType(enum.Enum):
    label = 1
    edittext = 2
    combobox = 3


class TkElement:

    def get_element(self):
        pass

    @staticmethod
    def init_element(master, type, value):
        if type == ColumnType.label:
            return Label(master, value)
        elif type == ColumnType.edittext:
            return EditText(master, value)
        elif type == ColumnType.combobox:
            return ComboBox(master, value)
        else:
            raise TypeError


class ComboBox(TkElement):
    def __init__(self, master, values, default_value_id=None):
        self.cv = tkinter.StringVar()
        self.com = ttk.Combobox(master, textvariable=self.cv)
        self.com["value"] = values

        if default_value_id is not None:
            self.com.current(default_value_id)

        self.com.bind("<<ComboboxSelected>>", self.func)

    def get_element(self):
        return self.com

    def func(self, event):
        print('Things to do on combo select value')
        pass


class EditText(TkElement):
    def __init__(self, master, text=''):
        self.element = tkinter.Entry(master, text=text)

    def get_element(self):
        return self.element


class Label(TkElement):
    def __init__(self, master, text=''):
        self.element = tkinter.Label(master, text=text)

    def get_element(self):
        return self.element


class Column:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value


class Row:
    def __init__(self, *row_elements):
        self.elements = row_elements


class Grid:
    def __init__(self, master, headers, columns):
        self.last_row = 0
        self.frame = tkinter.Frame(master)
        self.columns = columns
        for idx, header in enumerate(headers):
            label = TkElement.init_element(self.frame, type=ColumnType.label, value=header.value)
            element = label.get_element()
            element.grid(row=self.last_row, column=idx)
        self.last_row += 1

        self.add_row()
        self.frame.pack()

    def add_row(self):
        row = [TkElement.init_element(self.frame, c.type, c.value) for c in self.columns]
        for idx, item in enumerate(row):
            element = item.get_element()
            element.grid(row=self.last_row, column=idx)
        self.last_row += 1

