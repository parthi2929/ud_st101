"""
1. Read the list of .ipynb files from list_ipynb.csv in same folder
2. For each file, create respective html 
3. Fix 100% width issue for SVG element (CSS) for each html
4. write html in current folder (could also write in origin folder)
"""

import nbformat
import nbconvert
import sys
import os
import ntpath

# 1. Read the list of .ipynb files from list_ipynb.csv in same folder
import csv
result_paths = []
with open('list_ipynb.csv') as csvfile:
    ipynb_reader = csv.reader(csvfile)
    for each_row in ipynb_reader:
        current_file =  each_row[0]
        # print(current_file)

        """
        2. For each file, create respective html 
        https://github.com/jupyter/nbconvert/issues/699
        """
        with open(current_file, 'rb') as nb_file:
            nb_contents = nb_file.read().decode('utf8')  

        # Convert using the ordinary exporter
        notebook = nbformat.reads(nb_contents, as_version=4)      
        inname = ntpath.basename(current_file)
        outpath = os.path.dirname(current_file)
        outname = inname.split('.ipynb')[0] + '.html'
        print("\nInput:{} \nOutput:{} \nPath:{}".format(inname, outname, outpath))
        outpath = os.path.join(outpath,outname)  # outputting in same folder as ipynb file
        result_paths.append(outpath)  # to write list of htmls in a file
        exporter = nbconvert.HTMLExporter()
        body, res = exporter.from_notebook_node(notebook)        

        # Create a list saving all image attachments to their base64 representations
        images = []
        for cell in notebook['cells']:
            if 'attachments' in cell:
                attachments = cell['attachments']
                for filename, attachment in attachments.items():
                    for mime, base64 in attachment.items():
                        images.append( [f'attachment:{filename}', f'data:{mime};base64,{base64}'] )

        # Fix up the HTML and write it to disk
        for itmes in images:
            src = itmes[0]
            base64 = itmes[1]
            body = body.replace(f'src="{src}"', f'src="{base64}"', 1)  

        
        with open(outpath, 'w') as output_file:
            output_file.write(body)   

        #print('{} is done'.format(each_row[0]))   

list_str = ''
for each_html in result_paths:
    list_str += each_html + '\n'

with open('list_html.csv','w') as output_listfile:
    output_listfile.write(list_str)