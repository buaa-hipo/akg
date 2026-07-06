import argparse
import os
import re
from pathlib import Path


DEFAULT_LOG_DIR = "/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/log/level3/"
# DEFAULT_LOG_DIR = "/mnt/lustre-client/zhangsongyang/akg_4/aikg/log_dir"
# DEFAULT_LOG_DIR = "/mnt/lustre-client/zhangzizheng/AIKG/save_data/ds_v4_pro/logs/level3"
THROUGHPUT_PATTERN = re.compile(r"Output Throughput:\s*([\d.]+)")
USAGE_METADATA_PATTERN = re.compile(r"usage_metadata:\s*\{[^}]*'input_tokens':\s*(\d+)[^}]*'output_tokens':\s*(\d+)")


def collect_throughputs(file_path):
    throughputs = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = THROUGHPUT_PATTERN.search(line)
            if not match:
                continue

            try:
                throughputs.append(float(match.group(1)))
            except ValueError:
                continue

    return throughputs


def collect_usage_metadata(file_path):
    total_input_tokens = 0
    total_output_tokens = 0
    count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = USAGE_METADATA_PATTERN.search(line)
            if not match:
                continue

            try:
                input_tokens = int(match.group(1))
                output_tokens = int(match.group(2))
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                count += 1
            except ValueError:
                continue

    return total_input_tokens, total_output_tokens, count


def numeric_prefix_key(path):
    match = re.match(r"(\d+)_", path.name)
    if match:
        return int(match.group(1)), path.name
    return float("inf"), path.name


def main():
    parser = argparse.ArgumentParser(
        description="Compute average Output Throughput and token usage for all .log files in a directory."
    )
    parser.add_argument(
        "--log_dir",
        default=DEFAULT_LOG_DIR,
        help="Directory containing .log files.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Error: log_dir '{log_dir}' does not exist or is not a directory.")
        return

    log_files = sorted(log_dir.glob("*.log"), key=numeric_prefix_key)
    if not log_files:
        print(f"No .log files found in {log_dir}.")
        return

    total_sum = 0.0
    total_count = 0
    files_with_throughput = 0
    
    total_input_tokens_all = 0
    total_output_tokens_all = 0
    files_with_metadata = 0
    metadata_entry_count = 0

    print(f"Log dir: {log_dir}")
    print(f"Total .log files: {len(log_files)}")
    print("")

    for log_file in log_files:
        throughputs = collect_throughputs(log_file)
        input_tokens, output_tokens, metadata_count = collect_usage_metadata(log_file)
        
        if throughputs:
            file_sum = sum(throughputs)
            file_count = len(throughputs)
            file_avg = file_sum / file_count

            total_sum += file_sum
            total_count += file_count
            files_with_throughput += 1

            throughput_info = f"throughput: count={file_count}, avg={file_avg:.2f} tokens/s"
        else:
            throughput_info = "throughput: N/A"
        
        if metadata_count > 0:
            total_input_tokens_all += input_tokens
            total_output_tokens_all += output_tokens
            files_with_metadata += 1
            metadata_entry_count += metadata_count
            
            metadata_info = f"tokens: input={input_tokens}, output={output_tokens}, entries={metadata_count}"
        else:
            metadata_info = "tokens: N/A"

        print(f"{log_file.name}: {throughput_info}, {metadata_info}")

    print("")
    print("=== Throughput Summary ===")
    print(f"Files with throughput entries: {files_with_throughput}/{len(log_files)}")
    print(f"Total throughput entries: {total_count}")
    if total_count == 0:
        print("Overall Average Output Throughput: N/A")
    else:
        total_avg = total_sum / total_count
        print(f"Overall Average Output Throughput: {total_avg:.2f} tokens/s")

    print("")
    print("=== Token Usage Summary ===")
    print(f"Files with usage_metadata entries: {files_with_metadata}/{len(log_files)}")
    print(f"Total usage_metadata entries: {metadata_entry_count}")
    print(f"Total input_tokens: {total_input_tokens_all}")
    print(f"Total output_tokens: {total_output_tokens_all}")


if __name__ == "__main__":
    main()