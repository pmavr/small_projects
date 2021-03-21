# Inventory-Management-Software-in-Python
Inventory Management software in Tkinter library in python
by haymant1998 at github.com, adapted for mssql server using pyodbc

This is GUI based inventory management software implemented in tkinter library of python. 

Functionalities implemented:
1) Stock Management: This is used to add the new items that are brought in the inventory. It can also be used to update the existing items that are available in inventory. e.g, sometimes quantities of the products needs to be modified. Delete functionality is also provided to remove the items that are discontinued or not brought due to any reason.
2) Billing Section: Billing section is also there that would create the invoice and save it in .txt format in generated_bill folder that is also provided in this repository. It is used to create an invoice using item no. Customer name and customer phone number are mandatory part to generate invoice that would have invoice id as a random number between 1 and 1000.

Pre-Requisites:
1) Checked with python 3.8.5.
2) Make sure you have pyodbc installed in you venv.

Steps to run:
1) Create a database named 'ims' on an sql server instance, using ims.sql script.
2) Change odbc connection string accordingly in each of the .py files
3) Run stock_login.py file from your terminal or command prompt.
4) Enter login id and password (default id is admin and password is admin)

---------------------------You are good to go -----------------------------------
