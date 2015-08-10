# Maybe we can just read the .bin file without the Java code

import fileinput
for line in fileinput.input():
    print(line)