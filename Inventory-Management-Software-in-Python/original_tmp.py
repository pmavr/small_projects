import tkinter

from tmp import Grid, Column, ColumnType


if __name__ == '__main__':
    root = tkinter.Tk()
    values1 = ("Heilongjiang", "Jilin", "Liaoning")
    values2 = ('Athens', 'London', 'Rome')

    my_headers = [
        Column(ColumnType.label, 'Label_column'),
        Column(ColumnType.label, 'Combo_column_1'),
        Column(ColumnType.label, 'Combo_column_2'),
        Column(ColumnType.label, 'EditTxt_column')
    ]
    my_columns = [
        Column(ColumnType.label),
        Column(ColumnType.combobox, values1),
        Column(ColumnType.combobox, values2),
        Column(ColumnType.edittext)
    ]

    grid = Grid(root, my_headers, my_columns)

    grid.add_row()

    root.mainloop()
