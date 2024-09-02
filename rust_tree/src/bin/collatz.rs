// Determine the length of the collatz sequence beginning at `n`.
// If n=1, terminate
// If n is even, divide by 2
// If n is odd, multiply by 3 and add 1
fn collatz_length(mut n: i32) -> u32 {

    let mut len: u32 = 1;
    while n > 1 {
        
        if (n % 2) == 0 {
            // is even
            n = n/2;
        } else {
            n = (n*3)+1;
        }
        len += 1;
    };
    
    return len;
}

#[test]
fn test_collatz_length() {
    assert_eq!(collatz_length(11), 15);
}

fn main() {
    let n: i32 = 8;
    let l: u32 = collatz_length(n);
    println!("n: {n:#?}"); // # adds pretty printing
    println!("length for n={}: {}", n, l);
}
