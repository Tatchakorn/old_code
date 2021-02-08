import mysql.connector
from mysql.connector import Error
import xml.etree.ElementTree as ET
from xml.dom import minidom


def compute_gpa(score):
    if score >= 90:     # A+
        return 4.5
    elif score >= 85:   # A
        return 4
    elif score >= 80:   # A-
        return 3.7
    elif score >= 77:   # B+
        return 3.3
    elif score >= 73:   # B
        return 3
    elif score >= 70:   # B-
        return 2.7
    elif score >= 67:   # C+
        return 2.5
    elif score >= 63:   # C
        return 2.3
    elif score >= 60:   # C-
        return 2
    elif score >= 50:   # D
        return 1
    else:
        return 0        # E


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

        cursor = connection.cursor()
        query = "SELECT * FROM score_data;"
        cursor.execute(query)

        root = ET.Element("score_data")

        for (student_id, xml_score, data_structure_score, algorithm_score, network_score) in cursor:
            sub_1 = ET.SubElement(root, "student", student_id=student_id)
            xml_class = ET.SubElement(sub_1, "xml_class", gpa=str(compute_gpa(xml_score))).text = str(xml_score)
            xml_class = ET.SubElement(sub_1, "data_structure", gpa=str(compute_gpa(data_structure_score))).text = str(data_structure_score)
            xml_class = ET.SubElement(sub_1, "algorithm", gpa=str(compute_gpa(algorithm_score))).text = str(algorithm_score)
            xml_class = ET.SubElement(sub_1, "network", gpa=str(compute_gpa(network_score))).text = str(network_score)

        # Save the xml file
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open("./xml/output.xml", "w") as f:
            f.write(xmlstr)

except Error as e:
    print("connection failure：", e)

finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("database connection closed")
