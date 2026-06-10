"""单独运行在线答题"""
import argparse
from main import run_online
from utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="在线答题")
    parser.add_argument("questions", type=str, help="题目文件路径 (JSON)")
    args = parser.parse_args()

    run_online(args.questions)


if __name__ == "__main__":
    main()
