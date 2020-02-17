"""
Usage:
    $ python leastsquare.py <.data file> <.train file>
Output:
    Write the results to output.labels
"""
import sys
import os
import random
import matrix as mtx


def usage():
    return 'usage: python %s <.data file><.train file>\n' % os.path.basename(sys.argv[0])


def makeXR(x_in, r_in):
    x = []
    r = []
    for i in range(len(x_in)):
        if i in r_in:
            x.append([1]+x_in[i])
            r.append([r_in[i]])
        else:
            continue
    return x, r


def gradient_descent(x, r, n=100000, eta=0.0001, theta=0.01):
    # n = 100000
    # eta = 0.0001
    # theta = 0.01

    err = 0;
    prev_err = 100000000000000

    n_dim = len(x[0])
    w = mtx.zeros_matrix(n_dim, 1)
    for i in range(n_dim):
        sign = random.random()
        if sign > 0.5:
            w[i][0] = random.random()/100.0
        else:
            w[i][0] = -random.random()/100.0

    for i in range(n):
        print(f"=========================== {i}===========================")
        # x : n x d+1
        # w : d+1 x 1
        # r : n x 1

        # output diff
        mul = mtx.matrix_multiply(x, w)  # n x 1
        d = mtx.matrix_subtraction(r, mul)  # n x 1
        mat_err = mtx.matrix_multiply(mtx.transpose(d), d)
        err = mat_err[0][0] / 2

        if abs(prev_err - err) <= theta:
            print("!!!!!!!!!!!!! Finished !!!!!!!!!!!!!!")
            print(f"Loss: ", abs(prev_err-err))
            break

        print(f"Loss: ", abs(err))

        # weight diff
        eta_delta = mtx.matrix_multiply(mtx.transpose(d), x)
        w_delta = mtx.matrix_multiply_with_val(eta_delta, eta)
        w = mtx.matrix_addition(w, mtx.transpose(w_delta))

        # replace previous error
        prev_err = err

    return w


def train(x, r, n=100000, eta=0.0001, theta=0.01):
    w = []
    try:
        print("***** Try matrix calculation *****")
        xt = mtx.transpose(x)
        xtx = mtx.invert_matrix(mtx.matrix_multiply(xt, x))
        w = mtx.matrix_multiply(mtx.matrix_multiply(xtx, xt), r)
    except ArithmeticError:
        print("***** Failed in matrix calculation *****")
        print("***** Start gradient descent  *****")
        try:
            w = gradient_descent(x, r, n, eta, theta)
            print("***** Finished gradient descent *****")
        except ArithmeticError:
            print("---------- Matrix operator error -------------")
            sys.exit(1)

    return w


def predict(weight, test_data):
    result = []
    for i in range(len(test_data)):
        x = [1] + test_data[i]
        h = mtx.dot_product(weight, mtx.transpose(x))
        if h > 0:
            result.append([i] + [1])
        else:
            result.append([i] + [0])

    return result


def main():
    # Check command-line arguments
    if len(sys.argv) < 3:
        print(usage())
        sys.exit(1)
        data = []
    fdata = open(sys.argv[1])  # open data file
    ftrain = open(sys.argv[2])  # open train file
    flabel = open("output.labels", "wt")

    # fdata = open("ionosphere/ionosphere.data")  # open data file
    # ftrain = open("ionosphere/ionosphere.trainlabels.0")  # open train file

    
    r_in = {}
    x_in = []
    test_data = []

    try:
        # Prepare training data
        for line in ftrain:
            if line == "\n" or line == "" or len(line.split(" ")) < 2:
                continue
            line = (line.strip("\n").split(" "))
            val = int(line[0])
            if val == 1:
                r_in[int(line[1])] = 1
            else:
                r_in[int(line[1])] = -1

        # print(r_in)

        for line in fdata:
            if line == "\n" or line == "":
                continue
            line = line.strip().strip("\n").split(" ")
            try:
                x_in.append([float(x) for x in line])
                test_data.append([float(x) for x in line])
            except ValueError:
                print(f"Invalid characters in file!")
                sys.exit(1)

    finally:
        fdata.close()
        ftrain.close()

    # make input x, r
    x, r = makeXR(x_in, r_in)
    print(x)
    print(r)

    # train and get weight by x, r
    weight = train(x, r, n=100000, eta=0.0001, theta=0.0001)
    print(f"Weight: ", weight)

    # predict
    result = predict(weight, test_data)
    for x in result:
        for c in x:
            flabel.write(f"{c} ")
        flabel.write("\n")

    flabel.close()
    print("Wrote label file as 'output.label'")


if __name__ == "__main__":
    main()
