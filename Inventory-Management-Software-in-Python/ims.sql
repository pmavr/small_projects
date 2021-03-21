create database ims

use ims

CREATE TABLE users (
  sr int NOT NULL PRIMARY KEY,
  username varchar(20) NOT NULL,
  pass varchar(20) NOT NULL
) 

CREATE TABLE stocks (
  item_no varchar(11) NOT NULL PRIMARY KEY,
  name varchar(50) NOT NULL,
  qty int NOT NULL,
  price int NOT NULL
)

CREATE TABLE customer (
  customer_id int NOT NULL PRIMARY KEY,
  customer_name varchar(50) NOT NULL,
  customer_phone int NOT NULL
)

CREATE TABLE cust_trans (
  customer_id int NOT NULL,
  customer_trans_id int NOT NULL PRIMARY KEY,
  data varchar(10) NOT NULL,
  amount int NOT NULL,
  inv_id int NOT NULL
) 

CREATE TABLE purchase_bill (
  purchase_id int NOT NULL PRIMARY KEY,
  balance int NOT NULL,
  date varchar(10) NOT NULL,
  mode int NOT NULL,
  amount int NOT NULL
)


CREATE TABLE purchase_stock (
  purchase_id int NOT NULL PRIMARY KEY,
  item_no int NOT NULL,
  qty int NOT NULL
)

CREATE TABLE sales_bill (
  inv_id int NOT NULL PRIMARY KEY,
  date varchar(10) NOT NULL,
  amount int NOT NULL,
  customer_id int NOT NULL
)

CREATE TABLE sales_stocks (
  inv_id int NOT NULL PRIMARY KEY,
  item_no int NULL,
  qty int NOT NULL
)

CREATE TABLE wholesaler (
  whole_id int NOT NULL PRIMARY KEY,
  whole_name varchar(50) NOT NULL,
  whole_phone int NOT NULL,
  whole_address varchar(100) NOT NULL
)

CREATE TABLE whole_trans (
  whole_id int NOT NULL PRIMARY KEY,
  purchase_id int NOT NULL,
  date varchar(10) NOT NULL,
  amount int NOT NULL,
  whole_trans int NOT NULL
)




INSERT INTO users (sr, username, pass) VALUES
(1, 'admin', 'admin')

INSERT INTO stocks (item_no, name, qty, price) VALUES
(254, 'candy', 144, 2),
(297, 'face wash', 1, 40),
(451, 'Face Wash', 1, 120),
(583, 'Liquid Wash', 4, 230),
(656, 'face wash', 2, 40),
(691, 'Hand Wash', 2, 130),
(713, 'Lipstick', 5, 240),
(782, 'Body Wash', 9, 200),
(810, 'def', 5, 100),
(904, 'Liquid Cleaner', 5, 500)

INSERT INTO customer (customer_id, customer_name, customer_phone) VALUES
(1, 'Haymant', 123),
(293, 'bjbj', 2535),
(510, 'harman sandhu', 365367),
(514, 'iad', 12332),
(666, 'uadn', 123),
(777, 'njfan', 123),
(815, 'Hayadm', 123),
(967, 'ffdfd', 3435),
(978, 'bhalla', 55)


INSERT INTO sales_bill (inv_id, date, amount, customer_id) VALUES
(1160, '2019-11-14', 660, 1),
(1361, '2019-11-21', 44, 293),
(1583, '2019-11-03', 132, 1),
(1839, '2019-11-02', 220, 514),
(2720, '2019-11-03', 517, 1),
(2866, '2019-11-22', 2, 510),
(2934, '2019-11-06', 264, 777),
(3435, '2019-11-22', 11, 967),
(5602, '2019-11-21', 220, 978),
(6804, '2019-11-21', 132, 815),
(6907, '2019-11-03', 132, 1),
(9474, '2019-11-03', 572, 1),
(9735, '2019-11-15', 264, 666)



