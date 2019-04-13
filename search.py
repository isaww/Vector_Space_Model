import sys
import argparse
from vsm import VSM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", default=False, action='store_true',
                        help="If specified, turn on the relevance feedback")
    parser.add_argument("-b", default=False, action='store_true',
                        help="Run best version")
    parser.add_argument("-i", metavar="file-of-queries",
                        help="File path of the queries", required=True)
    parser.add_argument("-o", metavar="out-file-results",
                        help="File path for the output file", required=True)
    parser.add_argument("-m", metavar="model-dir",
                        help="File path for the model directory", required=True)
    parser.add_argument("-d", metavar="NTCIR-docs",
                        help="File path for the NTCIR-docs", required=True)
    args = parser.parse_args()


    if args.r == True or args.b == True:
        rel_feedback = True
    else:
        rel_feedback = False

    import time
    start = time.time()
    print ('start')

    test = VSM()
    test.read_inverted(args.m+'/inverted-file')
    test.vocab_to_index(args.m+'/vocab.all')
    test.doc_to_index(args.m+'/file-list')
    print ('processing docs...')
    test.process_docs(args.d)
    test.do_searching(args.i, args.o, rel_feedback) 

    print ('time: ', time.time() - start)


if __name__ == "__main__":
    main()
    
