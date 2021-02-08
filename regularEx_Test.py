import re
import os

print(os.listdir('C:\Program Files (x86)'))

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''

pattern = re.compile(r'abc')
matches = pattern.finditer(text_to_search)

# print raw string
print(r'\tTab')

# return an iterator for each match
for match in matches:
    print("First Ex.:", match)
print("Fist Ex. text pos: ", text_to_search[1:4])

# Using escape char for special chars
pattern = re.compile(r'coreyms\.com')
matches = pattern.finditer(text_to_search)
for match in matches:
    print("Second Ex.:", match)

# # ----Documentation ---- ##
# with open('./text/snippet.txt', 'r') as f:
#      Doc = f.read()
# print(Doc)
# # --- --- --- --- ---  --- ##

# print(read_data)
# pattern = re.compile(r'[1-5]\d\d[-.*]\d\d\d[-.*]\d\d\d')
# pattern = re.compile(r'[a-b]')
# pattern = re.compile(r'[^a-zA-Z]')
# pattern = re.compile(r'\d{3}.\d{3}.\d{4}')
# pattern = re.compile(r'Mr\.?\s[A-Z]\w*')

pattern = re.compile(r'(Mr|Ms|Mrs)\.?\s[A-Z]\w*')
matches = pattern.finditer(text_to_search)

for match in matches:
    print("Third Ex.:", match)


# Regular Expression for emails
regex_Email = r'[A-Za-z0-9_.-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+'
pattern = re.compile(regex_Email)
for match in pattern.finditer(emails):
    print("Fourth Ex.:", match)

# Regular Expression for urls
regex_Url = r'(https?://)?(www\.)?(\w+)(\.\w+)'
pattern = re.compile(regex_Url)
subbed_urls = pattern.sub(r'\3\4', urls) # replace the string with each group
print(subbed_urls)
for match in pattern.finditer(urls):
    print("Fifth Ex.(1):", match.group(0))  # The entire urls
    print("Fifth Ex.(2):", match.group(1))
    print("Fifth Ex.(3):", match.group(2))

for match in pattern.findall(urls):
    print(match)

print(pattern.search(urls))

# Flag
sentence = 'Start a sentence and then bring it to an end'
pattern = re.compile(r'start', re.IGNORECASE)
matches = pattern.search(sentence)
print(matches)

with open('./text/data.txt','r', encoding='utf-8') as f:
    read_data = f.read()

pattern = re.compile(regex_Email)
for match in pattern.findall(read_data):
    print(match)