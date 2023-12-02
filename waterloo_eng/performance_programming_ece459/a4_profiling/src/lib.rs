#![warn(clippy::all)]
pub mod checksum;
pub mod idea;
pub mod package;
pub mod student;

use idea::Idea;
use package::Package;

use std::io::prelude::*;
use std::fs;

pub enum Event {
    // Newly generated idea for students to work on
    NewIdea(Idea),
    // Termination event for student threads
    OutOfIdeas,
    // Packages that students can take to work on their ideas
    DownloadComplete(Package),
}

pub fn get_packages(num_pkgs: usize, file_path: &String) -> Vec<String> {

    let file = std::fs::File::open(file_path).unwrap();
    let reader = std::io::BufReader::new(file);
    let pkg_vec = reader.lines().collect::<Result<Vec<_>, _>>().unwrap();
    let pkg_vec_len = pkg_vec.len();
    
    // If the number of packages requested is larger than the amount in the file, pack an array
    // with that amount 
    if num_pkgs > pkg_vec_len {
        let cycle_iter = pkg_vec.iter().cycle();
        let pkg_vec_packed: Vec<String> = cycle_iter.take(num_pkgs).cloned().collect(); 

        return pkg_vec_packed;
    }

    return pkg_vec;
}

pub fn generate_ideas_vec(prod_file: &String, cust_file: &String) -> Vec<(String, String)> {

    let products = fs::read_to_string(prod_file).expect("file not found");
    let customers = fs::read_to_string(cust_file).expect("file not found");
    let ideas = cross_product(products, customers);

    return ideas;
}

fn cross_product(products: String, customers: String) -> Vec<(String, String)> {
    products
        .lines()
        .flat_map(|p| customers.lines().map(move |c| (p.to_owned(), c.to_owned())))
        .collect()
}
