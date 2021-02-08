import mysql.connector
from mysql.connector import Error
import xml.etree.ElementTree as ET
try:
    # Connect MySQL/test1 database
    connection = mysql.connector.connect(
        host='localhost', # host name
        database='xmldb', # database name
        user='tata',       # account
        password='123123')    # password

    if connection.is_connected():
        # version of database
        db_Info = connection.get_server_info()
        print("version of database：", db_Info)

        # show information of the current database
        cursor = connection.cursor()
        cursor.execute("SELECT DATABASE();")
        record = cursor.fetchone()
        print("current database：", record)

        sql = "INSERT INTO score_data (student_id, xml_score, data_structure_score, algorithm_score, network_score) " \
              "VALUES (%s, %s, %s, %s, %s);"

        # Read xml file
        tree = ET.parse("./xml/score_data.xml")
        root = tree.getroot()

        for student in root.iter('student'):
            temp = []
            for subject in student.iter():

                # Get grades
                if subject.text.isdigit():
                    temp.append(int(subject.text))

            # Attributes stored in a dictionary
            new_data = (student.attrib["student_id"], temp[0], temp[1], temp[2], temp[3])
            cursor = connection.cursor()
            cursor.execute(sql, new_data)

        # confirm data is stored into database
        connection.commit()

except Error as e:
    print("connection failure：", e)

finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("database connection closed")
