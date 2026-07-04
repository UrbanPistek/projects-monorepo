// TODO: Change from supporting generic types <T> to just i32
use std::cmp::Ordering;
use std::fmt::Debug;
use std::collections::VecDeque;

// Define additional data types to
// support the tree structure
// Point to store coordinates
struct Point<f32> {
    x: f32,
    y: f32,
}

// Box to store a rectangle
struct Box {
    bl: Point<f32>, // Bottom left point
    tr: Point<f32>, // Top right point
}

// "Branches" hold the MBR and pointers to the children
// "Leafs" hold the actual data
struct Node<const s: usize> {
    level: i32,
    mbr: Box<f32>,
    value: Box<f32>, // Make this more generic but use a box for now
    childern: [Node<s>; s], // If a leaf, children is empty
}

// Define attributes of the tree
struct RtreeConfig {
    node_max_childern: u32,
}

// Define the tree itself
type Rtree<const node_max_childern: usize> = Node<node_max_childern>;