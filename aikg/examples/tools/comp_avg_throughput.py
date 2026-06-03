import os
import re
import sys

def calculate_average_throughput(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return None
    
    throughputs = []
    pattern = r'Output Throughput: ([\d.]+)'
    
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                try:
                    throughput = float(match.group(1))
                    throughputs.append(throughput)
                except ValueError:
                    continue
    
    if not throughputs:
        print("No 'Output Throughput' entries found in the file.")
        return None
    
    average = sum(throughputs) / len(throughputs)
    return average, len(throughputs)

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python comp_avg_throughput.py <log_file_path>")
    #     sys.exit(1)
    
    # file_path = sys.argv[1]
    file_path = "/mnt/lustre-client/zhangzizheng/AIKG/akg/aikg/examples/log/level2/97_Matmul_BatchNorm_BiasAdd_Divide_Swish.log"
    result = calculate_average_throughput(file_path)
    
    if result:
        avg, count = result
        print(f"Total entries found: {count}")
        print(f"Average Output Throughput: {avg:.2f} tokens/s")