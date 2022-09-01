import argparse
parser = argparse.ArgumentParser(description='paddleocr_test')
parser.add_argument('--stage', type=str, default = 'infer', help='model stage')
parser.add_argument('--model', type=str, default='table', help='model name')
args = parser.parse_args()
print(args .stage)


