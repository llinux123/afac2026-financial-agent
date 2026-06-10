"""单独运行离线 Pipeline"""
from main import run_offline
from utils.logger import logger
import argparse


def main():
    parser = argparse.ArgumentParser(description="离线 Pipeline")
    parser.add_argument("--skip-summary", action="store_true",
                        help="跳过摘要生成")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="摘要生成并发线程数（默认6，范围1-16，超出自动修正）")
    args = parser.parse_args()

    run_offline()

    if not args.skip_summary:
        from main import run_summarize
        run_summarize(max_workers=args.max_workers)


if __name__ == "__main__":
    main()
