fn transpose(matrix: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    
    let m = matrix.len();
    let n = matrix[0].len();
    
    // Initialize empty
    // Shape needs to be inverted to handle a rectangular matrix as well 
    let mut transposed: Vec<Vec<i32>> = vec![vec![0; m]; n];
    for i in 0..n {
        for j in 0..m {
            transposed[i][j] = matrix[j][i];
        }
    }
    return transposed;
}

#[test]
fn test_transpose_3d() {
    let matrix: Vec<Vec<i32>> = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let transposed = transpose(&matrix);
    let output: Vec<Vec<i32>> = vec![
        vec![1, 4, 7],
        vec![2, 5, 8],
        vec![3, 6, 9],
    ];
    assert_eq!(transposed, output);
}

#[test]
fn test_transpose_2d() {
    let matrix: Vec<Vec<i32>> = vec![
        vec![1, 2],
        vec![4, 5],
    ];
    let transposed = transpose(&matrix);
    let output: Vec<Vec<i32>> = vec![
        vec![1, 4],
        vec![2, 5],
    ];
    assert_eq!(transposed, output);
}

#[test]
fn test_transpose_rect() {
    let matrix: Vec<Vec<i32>> = vec![
        vec![1, 2, 7],
        vec![4, 5, 8],
    ];
    let transposed = transpose(&matrix);
    let output: Vec<Vec<i32>> = vec![
        vec![1, 4],
        vec![2, 5],
        vec![7, 8]
    ];
    assert_eq!(transposed, output);
}

fn main() {
    let matrix: Vec<Vec<i32>> = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];

    println!("matrix: {:?}", matrix);
    let transposed = transpose(&matrix);
    println!("transposed: {:?}", transposed);
}
