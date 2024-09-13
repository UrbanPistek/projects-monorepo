use std::cmp::Ordering;

// Define type which is a pointer to a node
type SubTree<T> = Option<Box<Node<T>>>;

// Define the nodes in the tree
#[derive(Debug)]
struct Node<T: Ord> {
    value: T,
    left: SubTree<T>,
    right: SubTree<T>,
}

// Define the BST Itself
#[derive(Debug)]
struct BST<T: Ord> {
    root: SubTree<T>,
    size: u32,
}

// Implement the Node struct methods
impl<T: Ord> Node<T> {
    fn new(value: T) -> Self {
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
impl<T: Ord> BST<T> {
    fn new(value: T) -> BST<T> {
        BST {
            root: Some(Box::new(Node::new(value))),
            size: 1,
        }
    }

    // Insert a value into the appropriate location in this tree using an iterative approach
    fn insert(&mut self, value: T) -> Result<InsertResult, InsertResult> {
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
}

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
}
