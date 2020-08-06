with open('../annotations/train.txt', 'r') as f:
    lines = f.readlines()

lines = [x.strip() for x in lines]
line = lines[0]
line.split('"')[3]
