"""
CITS4404 Group C1

Script that processes the entire Friends TV show dialogue and extracts questions and answers.

"""


def read_lines(character='JOEY', filename=None):
    # replace file path
    with open("/Users/EleanorLeung/Documents/CITS4404/chatbot/data/friends-final.txt", 'r') as f:
        lines = f.readlines()

    questions, answers = [], []
    for index in range(0, len(lines) - 1):
        line = lines[index].split("\t")
        if len(line) == 8:
            if line[2] == character:
                question = lines[index - 1].split("\t")[5]
                answer = line[5]
                questions.append(question)
                answers.append(answer)

    assert(len(questions) == len(answers))
    return questions, answers


def write_to_file(questions=None, answers=None, filename=None):
    # replace file path
    filename = "/Users/EleanorLeung/Documents/CITS4404/chatbot/friends_corpus/friends_data/cleaned_corpus.txt"

    with open(filename, 'wb') as fw:
        for index in range(0, len(questions) - 1):
            fw.write(str.encode(questions[index]) + b"\n")
            fw.write(str.encode(answers[index]) + b"\n")


if __name__ == '__main__':
    questions, answers = read_lines()
    write_to_file(questions=questions, answers=answers)