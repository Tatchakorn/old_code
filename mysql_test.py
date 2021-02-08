import mysql.connector
from mysql.connector import Error

try:
    # Connect MySQL/test1 database
    connection = mysql.connector.connect(
        host='localhost', # host name
        database='classmate', # database name
        user='Ta',        # account
        password='123456')  # password

    if connection.is_connected():
        # version of database
        db_Info = connection.get_server_info()
        print("version of database：", db_Info)

        # show information of the current database
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")
        record = cursor.fetchone()
        print("current database：", record)
except Error as e:
    print("connection failure：", e)

finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("database connection closed")
