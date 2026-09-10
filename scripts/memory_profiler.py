import time
import subprocess
import psutil

def main():
    print("[MEMORY PROFILER] Started background memory profiler.", flush=True)
    while True:
        try:
            mem = psutil.virtual_memory()
            
            ps_output = subprocess.check_output(
                ["ps", "-eo", "pid,rss,cmd", "--sort=-rss"], 
                text=True
            )
            lines = ps_output.strip().split('\n')
            
            # Format: PID RSS(KB) CMD
            top_procs = " | ".join([f"{l.split()[1]} KB: {l.split()[2].split('/')[-1]}" for l in lines[1:4]])
            
            print(f"[MEMORY PROFILER] RAM Used: {mem.percent}% | Top: {top_procs}", flush=True)
        except Exception:
            pass
        time.sleep(2)

if __name__ == "__main__":
    main()
