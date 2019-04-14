# Vector Space Model
## Features
- Tf-idf
- Rocchio feedback
## Options
    parser.add_argument("-r", default=False, action='store_true',
                        help="If specified, turn on the relevance feedback")
    parser.add_argument("-i", metavar="file-of-queries",
                        help="File path of the queries", required=True)
    parser.add_argument("-o", metavar="out-file-results",
                        help="File path for the output file", required=True)
    parser.add_argument("-m", metavar="model-dir",
                        help="File path for the model directory", required=True)
    parser.add_argument("-d", metavar="NTCIR-docs",
                        help="File path for the NTCIR-docs", required=True)
                        
## Run
python3 search.py
