import os
import time

def main():
    ts = time.perf_counter()
    
    # Do something 
    time.sleep(1)

    te = time.perf_counter()
    print(f"Completed in {round(te-ts,3)}s")

main()
