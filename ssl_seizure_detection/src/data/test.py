import multiprocessing
import time

def cpu_task(n):
    """Simulates a CPU-intensive task by calculating the sum of squares."""
    print(f"Process {multiprocessing.current_process().name} started")
    total = sum(i**2 for i in range(n))
    print(f"Process {multiprocessing.current_process().name} finished")
    return total

if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()  # Get available CPU cores
    print(f"Using {num_cores} CPU cores")

    numbers = [10**6] * num_cores  # Workload for each core

    start_time = time.time()
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(cpu_task, numbers)  # Distribute workload
    
    end_time = time.time()

    print(f"Results: {results}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

