"""
CITS4404 Group C1
Cleans the twitter data
"""

import os
import re

from datetime import datetime

def read_raw_data(source_path='./twitter_data'):
    lines = []
    files = ["%s/%s" % (source_path, f) for f in os.listdir(source_path)]
    for f in files:
        lines += open(f, 'rt').readlines()
    return lines

def clean_line(line):
    """ Remove links, tags, multiple spaces, tabs and newlines """
    line = re.sub(r"http\S+", "", line)
    line = re.sub(r"\@[a-z0-9-_][a-z0-9-_]*", '', line)
    line = re.sub(r"\#[a-z0-9-_][a-z0-9-_]*", '', line)
    line = re.sub("\s+", ' ', line).strip()

    return line

def write_to_file(raw_lines, filename=None):
    filename = "./twitter_data/cleaned_corpus.txt"

    with open(filename, 'wb') as fw:
        for id in range(len(raw_lines)//2):
            question = clean_line(raw_lines[2 * id])
            answer = clean_line(raw_lines[2 * id + 1])

            if (len(question) > 10) and (len(answer) > 10):
                fw.write(str.encode(question) + b"\n")
                fw.write(str.encode(answer) + b"\n")
            if (id % 1000) == 0:
                print("Cleaned %i lines @ %s" % (id, datetime.now()))


if __name__ == '__main__':
    raw_lines = read_raw_data()
    write_to_file(raw_lines)
