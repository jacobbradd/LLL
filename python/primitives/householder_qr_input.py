from householder import *
import ast
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("-f", "--outsideqr", action="store_true")
parser.add_argument("-t", "--timeit", action="store_true")


def main():
    args = parser.parse_args()
    B = np.array(ast.literal_eval(input()))
    #np.set_printoptions(threshold=np.inf)
    start = time.perf_counter()
    if args.outsideqr:
        R = np.linalg.qr(B, mode="r")
    else:    
        R = qr_wy(B, r_mode = True)
    if not args.quiet:
        print(R)
    end = time.perf_counter()
    if args.timeit:
        print("Time elapsed:", end-start, "seconds.")
    
main()