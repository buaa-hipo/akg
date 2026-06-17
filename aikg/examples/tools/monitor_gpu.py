import subprocess
import time

def get_gpu1_process_count():
    """获取GPU1上的活跃进程数量"""
    try:
        # 执行nvidia-smi命令获取进程信息
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        output = result.stdout.strip()
        
        # 统计GPU1上的进程数
        gpu1_process_count = 0
        if output:
            lines = output.split('\n')
            for line in lines:
                # 每行格式: GPU-uuid, pid
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_uuid = parts[0].strip()
                    # 检查是否是GPU1（通过索引判断）
                    result_device = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                        capture_output=True,
                        text=True
                    )
                    device_info = result_device.stdout.strip()
                    for device_line in device_info.split('\n'):
                        device_parts = device_line.split(',')
                        if len(device_parts) >= 2:
                            gpu_index = device_parts[0].strip()
                            device_uuid = device_parts[1].strip()
                            if device_uuid == gpu_uuid and gpu_index == '1':
                                gpu1_process_count += 1
        return gpu1_process_count
    except Exception as e:
        print(f"获取进程信息失败: {e}")
        return 0

def write_nvidia_smi_to_log():
    """将nvidia-smi输出写入日志文件"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        with open('monitor_gpu.log', 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write('='*60 + '\n')
            f.write(result.stdout)
            f.write('\n')
        print(f"GPU1活跃进程>=2，已记录到monitor_gpu.log")
    except Exception as e:
        print(f"写入日志失败: {e}")

def main():
    """主函数：每隔2秒检测GPU1进程"""
    print("开始监控GPU1进程，每隔2秒检测一次...")
    print("当GPU1活跃进程>=2时，将nvidia-smi输出写入monitor_gpu.log")
    
    try:
        while True:
            process_count = get_gpu1_process_count()
            # print(f"GPU1活跃进程数: {process_count}", end='\r')
            
            if process_count >= 2:
                write_nvidia_smi_to_log()
            
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()