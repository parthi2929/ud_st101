"""
1. Read the list of .html files from list_html.csv in same folder
2. upload each file to wp and get back the ID
3. write the IDs (urls?) of the uploaded files as list_wpurls.csv
4. UPLOAD SAME FILE ONLY ONCE

thanks: https://stackoverflow.com/questions/43915184/how-to-upload-images-using-wordpress-rest-api-in-python
"""

import sys
import os
import ntpath
import getpass
import requests
import pprint

url='http://rparthiban.com/articles/wp-json/wp/v2/media'


import csv
result_paths = []

user = input('Username:')
password = getpass.getpass('Password:')


session = requests.Session()
with open('list_html.csv') as csvfile:
    ipynb_reader = csv.reader(csvfile)
    for each_row in ipynb_reader:
        current_file =  each_row[0]
        # print(current_file)

        data = open(current_file, 'rb').read()
        fileName = os.path.basename(current_file)

        res = requests.post(url,
                            data=data,
                            headers={ 'User-Agent': 'Mozilla/5.0', 'Content-Type': '','Content-Disposition' : 'attachment; filename=%s'% fileName},
                            auth=(user, password))
        pp = pprint.PrettyPrinter(indent=4) ## print it pretty. 
        pp.pprint(res.json()) #this is nice when you need it
        # newDict=res.json()
        # newID= newDict.get('id')
        # link = newDict.get('guid').get("rendered")
        # print(newID, link)
