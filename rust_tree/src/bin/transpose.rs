fn transpose(matrix: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    
    let m = matrix.len();
    let n = matrix[0].len();
    
    // Initialize empty
    let mut transposed: [[i32; 3]; 3] = [[0; 3]; 3];
    for i in 0..m {
        for j in 0..n {
            transposed[i][j] = matrix[j][i];
        }
    }

    return transposed;
}

#[test]
fn test_transpose() {
    let matrix = [
        [101, 102, 103], //
        [201, 202, 203],
        [301, 302, 303],
    ];
    let transposed = transpose(matrix);
    assert_eq!(
        transposed,
        [
            [101, 201, 301], //
            [102, 202, 302],
            [103, 203, 303],
        ]
    );
}

fn main() {
    let matrix = [
        [101, 102, 103], // <-- the comment makes rustfmt add a newline
        [201, 202, 203],
        [301, 302, 303],
    ];

    println!("matrix: {:#?}", matrix);
    let transposed = transpose(matrix);
    println!("transposed: {:#?}", transposed);
}
