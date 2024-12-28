// TODO: Change from supporting generic types <T> to just i32
use std::cmp::Ordering;
use std::fmt::Debug;
use std::collections::VecDeque;

// Define type which is a pointer to a node
#[allow(non_camel_case_types)]
type SubTree<i32> = Option<Box<Node<i32>>>;

// Define the nodes in the tree
#[derive(Debug)]
#[allow(non_camel_case_types)]
struct Node<i32> {
    value: i32,
    left: SubTree<i32>,
    right: SubTree<i32>,
}

// Define the BST Itself
#[derive(Debug)]
#[allow(non_camel_case_types)]
struct BST<i32> {
    root: SubTree<i32>,
    size: u32,
}

// Implement the Node struct methods
impl Node<i32> {
    fn new(value: i32) -> Self {
        Node {
            value,
            left: None,
            right: None,
        }
    }
}

// Used to indicate status of insert
#[derive(Debug, PartialEq)]
enum InsertResult {
    Success,
    Duplicate,
    Failure,
}

// Define all node-based functions
impl BST<i32> {
    fn new(value: i32) -> BST<i32> {
        BST {
            root: Some(Box::new(Node::new(value))),
            size: 1,
        }
    }

    // Insert a value into the appropriate location in this tree using an iterative approach
    fn insert(&mut self, value: i32) -> Result<InsertResult, InsertResult> {
        // Start by referencing the root node of the tree
        let mut current = &mut self.root;

        // Traverse the tree iteratively until an appropriate spot for the new value is found
        while let Some(ref mut node) = current {
            match value.cmp(&node.value) {

                // If value is less than the current node's value, move to the left subtree
                Ordering::Less => {
                    if node.left.is_none() {
                        node.left = Some(Box::new(Node::new(value)));
                        self.size += 1;
                        return Ok(InsertResult::Success);
                    }
                    current = &mut node.left;
                }

                // If value is greater than the current node's value move to the right subtree
                Ordering::Greater => {
                    if node.right.is_none() {
                        node.right = Some(Box::new(Node::new(value)));
                        self.size += 1;
                        return Ok(InsertResult::Success);
                    }
                    current = &mut node.right;
                }

                // If it is equal, its a duplicate, skip but indicate this in the return
                Ordering::Equal => {
                    // If the value is equal, we don't insert it (to avoid duplicates)
                    return Err(InsertResult::Duplicate);
                }
            }
        }

        // Report a failure
        Err(InsertResult::Failure)
    }

    // Iteratively find the height of the tree
    fn height(&mut self) -> Result<u32, ()> {
    
        // Base case, its only the root
        if self.size == 1 {
            return Ok(1);
        }

        // Create queue for breath-first traversal
        // Store address of the subtree
        let mut queue: VecDeque<&SubTree<i32>> = VecDeque::new();

        // Initialize
        let mut height: u32 = 0;
        queue.push_front(&self.root);
        
        // Main loop
        loop {

            // Increment height
            let mut node_count = queue.len();
            if node_count == 0 { break };
                
            height += 1;
        
            // Empty all items in the queue
            // on the current layer
            while node_count > 0 {

                // Explicitly unwrap this
                let node_ref = match queue.pop_front() {
                    Some(val) => val, // Return the reference 
                    None => &None
                }; 
                let node = node_ref.as_ref().unwrap();

                // Push reference to next nodes if they exist
                if !node.left.is_none(){
                    queue.push_front(&node.left);
                }
                if !node.right.is_none(){
                    queue.push_front(&node.right);
                }
                
                node_count -= 1;
            }
            
        }
        
        return Ok(height);

    }
}

// Needs to be defined externally
// Helper function to get matrix form of a bst
fn get_bst_matrix_recursive (matrix: &mut Vec<Vec<i32>>, root: SubTree<i32>, col: usize, row: usize, height: u32) -> () {

    let node = *(root.unwrap());

    // Store the value of the node
    matrix[row][col] = node.value;

    // Recursive base case
    if node.left.is_none() && node.right.is_none() {
        return
    }

    // Recurse
    let col_inc = 2_u32.pow(height-2);
    let l = col-col_inc as usize;
    let r = col+col_inc as usize;
    get_bst_matrix_recursive(matrix, node.left, l, row+1, height-1);
    get_bst_matrix_recursive(matrix, node.right, r, row+1, height-1);

}

// Pretty display of a bst
fn print_bst_matrix(bst_matrix: Vec<Vec<i32>>) {
    // Get dimensions of the matrix
    let m = bst_matrix.len();
    let n = bst_matrix[0].len();

    for i in 0..m {
        for j in 0..n {
            // Attempt to downcast to i32, u32, etc.
            if let Some(&value) = bst_matrix[i].get(j) {
                if value == 0 {
                    print!(" ");  // Print space for 0
                } else {
                    print!("{:?}", value);  // Print the integer value
                }
            } else {
                return
            }
        }
        println!();
    }
}

// // Helper function to get matrix form of a bst
// fn print_bst (bst: &mut BST<i32>) -> () {

//     let h: u32 = bst.height().unwrap();
//     let cols = 2_u32.pow(h) - 1;
//     let rows = h as usize;
//     let m = rows;
//     let n = cols as usize;
//     let mut matrix: Vec<Vec<i32>>= vec![vec![0; n]; m];
//     println!("initial matrix[{:?}, {:?}]: {:?}", m, n, matrix);

//     // Get the matrix form of the tree
//     let mid: f32 = (cols as f32)/2.0;
//     let i = mid.floor() as usize;
//     get_bst_matrix_recursive(&mut matrix, bst.root, i, 0, h);
//     println!("bst matrix: {:?}", matrix);

//     println!("bst pretty print: ");
//     print_bst_matrix(matrix);
// }

// Unit tests for the BST
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_success() {
        let mut bst = BST::new(10);
        let res = match bst.insert(5) {
            Ok(InsertResult::Success) => InsertResult::Success,
            Err(InsertResult::Duplicate) => InsertResult::Duplicate,
            Err(InsertResult::Failure) => InsertResult::Failure,
            _ => panic!() // Covers all other branches
        };
        assert_eq!(res, InsertResult::Success); // Should succeed
        assert_eq!(bst.size, 2); // Tree size should be 1 (initial + 1 inserts)
    }

    #[test]
    fn test_height_1() {

        let mut bst = BST::new(10);
        assert_eq!(bst.height().unwrap(), 1);
    }

    #[test]
    fn test_height_2() {

        let mut bst = BST::new(10);
        bst.insert(5).unwrap();
        assert_eq!(bst.height().unwrap(), 2);
    }
    
    #[test]
    fn test_height_3() {

        let mut bst = BST::new(10);
        bst.insert(5).unwrap();
        bst.insert(15).unwrap();
        bst.insert(3).unwrap();
        bst.insert(7).unwrap();
        assert_eq!(bst.height().unwrap(), 3);
    }

    #[test]
    fn test_height_4() {

        let mut bst = BST::new(10);
        bst.insert(5).unwrap();
        bst.insert(15).unwrap();
        bst.insert(3).unwrap();
        bst.insert(7).unwrap();
        bst.insert(6).unwrap();
        assert_eq!(bst.height().unwrap(), 4);
    }

}

fn main() {
    let mut bst = BST::new(10);
    bst.insert(5).unwrap();
    bst.insert(15).unwrap();
    bst.insert(3).unwrap();
    bst.insert(7).unwrap();

    // This is inserting a duplicate and should throw a message
    match bst.insert(5) {
        Ok(InsertResult::Success) => println!("Insert successful"),
        Err(InsertResult::Duplicate) => println!("Value already exists in the tree"),
        Err(InsertResult::Failure) => println!("Insert failed for some reason"),
        _ => panic!() // Covers all other branches
    }
    
    println!("{:#?}", bst);
    println!("bst size: {:?}", bst.size);
    println!("bst height: {:?}", bst.height().unwrap());

    // test show
    let h: u32 = bst.height().unwrap();
    let cols = 2_u32.pow(h) - 1;
    let rows = h as usize;
    let m = rows;
    let n = cols as usize;
    let mut matrix: Vec<Vec<i32>>= vec![vec![0; n]; m];
    println!("initial matrix[{:?}, {:?}]: {:?}", m, n, matrix);

    // Get the matrix form of the tree
    let mid: f32 = (cols as f32)/2.0;
    let i = mid.floor() as usize;
    get_bst_matrix_recursive(&mut matrix, bst.root, i, 0, h);
    println!("bst matrix: {:?}", matrix);

    println!("bst pretty print: ");
    print_bst_matrix(matrix);
}
