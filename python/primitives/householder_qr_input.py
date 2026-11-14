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
    # Warm-up
    if args.outsideqr:
        np.linalg.qr(B, mode="r")
    else:
        qr_wy(B, r_mode=True)
    start = time.perf_counter()
    if args.outsideqr:
        R = np.linalg.qr(B, mode="r")
    else:    
        R = qr_wy(B, r_mode = True)
    end = time.perf_counter()
    if not args.quiet:
        np.set_printoptions(threshold=np.inf)
        print(R)
    if args.timeit:
        print("Time elapsed:", end-start, "seconds.")
    
main()