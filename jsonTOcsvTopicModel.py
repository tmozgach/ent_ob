# Python 3
# The following script takes the text of posts and title from Reddit for Topic Modeling analysis
# Run inside the folder that contains your JSON files.
# Every row of that csv  is one thread (title, post, comments).

import json
import re
import glob
import fileinput
import csv

for filename in glob.glob('*.json'):
    with open(filename, 'r') as f:
        for line in f:

            # The title is inside "title" field
            title = re.findall(r'\"title\": \"(.*?)\",',line)
            # The main post is inside "selftext" field
            post = re.findall(r'\"selftext\": \"(.*?)\",',line) 
            # The comment is inside "body" field
            comments = re.findall(r'\"body\": \"(.*?)\",',line)
           
            text = ' '.join([str(e) for e in title])
            text = text + ' '
            text = text + ' '.join(str(e) for e in post)
            text = text + ' '
            text = text + ' '.join(str(e) for e in comments) 

            # Write to CVS file
            with open("output.csv", "a+", newline='') as output:
                writer = csv.writer(output)
                writer.writerow([text])
                #writer.writerow(post)
                #for comment in comments:
                #    if comment != "[deleted]":
                #       writer.writerow([comment])

