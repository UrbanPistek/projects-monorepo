fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

fn bubble_sort(arr: &mut [i32]) {
    let len = arr.len();
    for i in 0..len {
        for j in 0..len - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
            }
        }
    }
}

fn string_operations(iterations: usize) -> String {
    let mut result = String::new();
    for i in 0..iterations {
        result.push_str(&format!("Iteration {}, ", i));
    }
    result
}

fn main() {

    // Recursive function (stack heavy)
    println!("Computing Fibonacci...");
    let fib_result = fibonacci(35);
    println!("Fibonacci(35) = {}", fib_result);
    
    // Sorting (CPU intensive)
    println!("Sorting array...");
    let mut numbers: Vec<i32> = (0..5000).rev().collect();
    bubble_sort(&mut numbers);
    println!("Sorted {} numbers", numbers.len());
    
    // String allocation (memory heavy)
    println!("String operations...");
    let text = string_operations(100000);
    println!("Generated string with {} characters", text.len());
}