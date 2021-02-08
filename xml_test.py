import xml.etree.ElementTree as ET
from yattag import Doc, indent


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

tree = ET.parse("./xml/score_data.xml")
root = tree.getroot()

doc, tag, text = Doc().tagtext()

# Self closing tags
with tag('score_data'):
    with tag('student'):
        with tag('field1', student_id='blah'):
            text('some value1')
            with tag('shit'):
                pass
            with tag('shit'):
                pass
        with tag('field2', student_id='asdfasd'):
            text('some value2')

# indent the result
result = indent(doc.getvalue(), indentation=' '*4, newline='\r\n', indent_text=True)
print(result)

# Compute GPA from xml file
"""
print(root.tag)
for student in root.iter('student'):
    # Attributes stored in a dictionary
    print("student_id\t", student.attrib["student_id"])
    for subject in student.iter():
        print(subject.tag, end=' ')

        # Get grades
        if subject.text.isdigit():
            print(subject.text)

for student in root.iter('student'):
    for subject in student.iter():
        if subject.text.isdigit():
            subject.set("gpa", str(compute_gpa(int(subject.text))))

tree.write("./xml/output.xml")
"""