# this program is set to take 4 arguments of the form:
#<training file> <testfile> alpha <number of iterations>


import sys
assert len(sys.argv) == 5, "Invalid number of arguments"

try:
    trainFile = sys.argv[1]
    testFile = sys.argv[2]
except ValueError:
    print("Invalid input files")
    raise SystemExit

try:
    alpha =int( sys.argv[3])
    numIterations = int(sys.argv[4])
except ValueError:
    print("alpha or number of iterations entered could",)
    print(" not be converted to integers")
    raise SystemExit

