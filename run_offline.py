"""单独运行离线 Pipeline"""
from main import run_offline, run_summarize
from utils.logger import logger
import argparse


def main():
    parser = argparse.ArgumentParser(description="离线 Pipeline")
    parser.add_argument("--skip-summary", action="store_true",
                        help="跳过摘要生成")
    args = parser.parse_args()

    run_offline()

    if not args.skip_summary:
        run_summarize()


if __name__ == "__main__":
    main()
