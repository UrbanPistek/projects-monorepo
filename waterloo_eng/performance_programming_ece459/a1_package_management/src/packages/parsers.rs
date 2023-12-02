use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use regex::Regex;

use crate::Packages;
use crate::packages::RelVersionedPackageNum;
use crate::packages::VersionRelation;
use crate::packages::Dependency;

use rpkg::debversion;

const KEYVAL_REGEX : &str = r"(?P<key>(\w|-)+): (?P<value>.+)";
const PKGNAME_AND_VERSION_REGEX : &str = r"(?P<pkg>(\w|\.|\+|-)+)( \((?P<op>(<|=|>)(<|=|>)?) (?P<ver>.*)\))?";

impl Packages {
    /// Loads packages and version numbers from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate value into the installed_debvers map with the parsed version number.
    pub fn parse_installed(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        
        if let Ok(lines) = read_lines(filename) {
            
            // Variables to store parsed values
            let mut current_package_num = 0;

            #[allow(unused_variables, unused_assignments)]
            let mut key: &str = "";

            #[allow(unused_variables, unused_assignments)]
            let mut value: &str = "";
            
            for line in lines {
                if let Ok(ip) = line {
                    
                    // run regex on the line to find the key and value pairs 
                    match kv_regexp.captures(&ip) {
                        None => (), 
                        Some(caps) => {
                            key = caps.name("key").unwrap().as_str();
                            value = caps.name("value").unwrap().as_str();

                            // Search for Appropiate Key
                            if key == "Package" {
                                current_package_num = self.get_package_num_inserting(&value); 
                            }
                            
                            if key == "Version" {
                                let version_value = value.trim().parse::<debversion::DebianVersionNum>().unwrap(); 

                                // Save 
                                self.installed_debvers.insert(current_package_num, version_value);
                            }
                        }
                    }
                }
            }
        }

        println!("Packages installed: {}", self.installed_debvers.keys().len());
    }

    /// Loads packages, version numbers, dependencies, and md5sums from a file, calling get_package_num_inserting on the package name
    /// and inserting the appropriate values into the dependencies, md5sum, and available_debvers maps.
    pub fn parse_packages(&mut self, filename: &str) {
        let kv_regexp = Regex::new(KEYVAL_REGEX).unwrap();
        let pkgver_regexp = Regex::new(PKGNAME_AND_VERSION_REGEX).unwrap();

        if let Ok(lines) = read_lines(filename) {
            
            let mut current_package_num = 0;

            #[allow(unused_variables, unused_assignments)]
            let mut key: &str = "";

            #[allow(unused_variables, unused_assignments)]
            let mut value: &str = "";
            
            for line in lines {
                if let Ok(ip) = line {
                    
                    match kv_regexp.captures(&ip) {
                        None => (), 
                        Some(caps) => {
                            key = caps.name("key").unwrap().as_str();
                            value = caps.name("value").unwrap().as_str();
                            
                            // Search for Appropiate Key
                            if key == "Package" {
                                current_package_num = self.get_package_num_inserting(&value); 
                            }
                            
                            if key == "Version" {
                                let version_value = value.trim().parse::<debversion::DebianVersionNum>().unwrap(); 

                                // Save 
                                self.available_debvers.insert(current_package_num, version_value);
                            }
                            
                            if key == "MD5sum" {
                                let hash_value = value.to_string();

                                // Save 
                                self.md5sums.insert(current_package_num, hash_value);
                            }
                            
                            if key == "Depends" {
                                let depends = value.to_string();
                                let individual_depends = depends.split(",");

                                // Top level dependancy vector to store all dependancies
                                let mut dependancy_vec: Vec<Dependency> = Vec::new();

                                for dep in individual_depends {

                                    // Create vector for each dependancy
                                    let mut alt_vec: Vec<RelVersionedPackageNum> = Vec::new();

                                    // get alternatives
                                    let alternatives = dep.split("|");
                                    for alt in alternatives {
                                    
                                        // perform regex to extract package info
                                        match pkgver_regexp.captures(&alt) {
                                            None => (),
                                            Some(alts) => {

                                                // explicitly unwrap the values to handle none
                                                // let pkg = alts.name("pkg").unwrap().as_str();
                                                let mut pkg_value: &str = "";
                                                
                                                #[allow(unused_variables, unused_assignments)]
                                                let mut pkg_num = 0;
                                                
                                                match alts.name("pkg") {
                                                    None => (),
                                                    Some(pkg) => {pkg_value = pkg.as_str()}
                                                }
                                                pkg_num = self.get_package_num_inserting(&pkg_value);
                                                
                                                let mut op_value: debversion::VersionRelation = debversion::VersionRelation::StrictlyGreater; 
                                                match alts.name("op") {
                                                    None => (),
                                                    Some(op) => {op_value = op.as_str().parse::<debversion::VersionRelation>().unwrap()}
                                                }
                                                
                                                let mut ver_value: String = String::from("0.0"); 
                                                match alts.name("ver") {
                                                    None => (),
                                                    Some(ver) => {ver_value= ver.as_str().to_string()}
                                                }

                                                // Define option type
                                                let rel_version_opt: Option<(VersionRelation, String)> = Some((op_value, ver_value));

                                                // Create alternate struct 
                                                let alt_struct = RelVersionedPackageNum {package_num: pkg_num, rel_version: rel_version_opt};

                                                // Push into the vector
                                                alt_vec.push(alt_struct);

                                            }
                                        }
                                    }

                                    // Push into top level vector
                                    dependancy_vec.push(alt_vec);
                                }

                                // Insert into dependacies hash map
                                self.dependencies.insert(current_package_num, dependancy_vec);
                            }
                        }
                    }
                }
            }
        }
        println!("Packages available: {}", self.available_debvers.keys().len());
    }
}


// standard template code downloaded from the Internet somewhere
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
