use std::collections::HashMap;
use std::collections::VecDeque;

use crate::Packages;
use crate::packages::Dependency;
use crate::packages::RelVersionedPackageNum;
use crate::packages::VersionRelation;
use rpkg::debversion; 

impl Packages {

    /// Computes a solution for the transitive dependencies of package_name; when there is a choice A | B | C, 
    /// chooses the first option A. Returns a Vec<i32> of package numbers.
    ///
    /// Note: does not consider which packages are installed.
    pub fn transitive_dep_solution(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }

        let mut deps : &Vec<Dependency> = &*self.dependencies.get(self.get_package_num(package_name)).unwrap();

        #[allow(unused_variables, unused_assignments)]
        let mut dependency_set = vec![];

        // Create hashmaps to cache and function as worklist
        let mut cache: HashMap<String, i32> = HashMap::new();
        let mut worklist: VecDeque<String> = VecDeque::new();

        // implement worklist
        // Insert initial list of dependancies to worklist
        self.transitive_add_deps_to_worklist(&mut worklist, &cache, deps);
        // println!("=> Initial Worklist len: {}", worklist.len());

        // continue until worklist is empty
        let mut curr_dep_num: i32;
        while !worklist.is_empty() {
           
            // get value from the front of the queue
            match worklist.pop_front() {
                None => (),
                Some(val) => {
                   
                    // Check if value is in cache
                    if !cache.contains_key(&val) {
                        
                        // Insert into cache if its not there
                        curr_dep_num = self.get_package_num(&val).clone();
                        cache.insert(val, curr_dep_num);

                        // Get deps
                        deps = &*self.dependencies.get(&curr_dep_num).unwrap();
                        self.transitive_add_deps_to_worklist(&mut worklist, &cache, deps);
                    }
                }
            }

            // For debugging
            // println!("Worklist length: {}", worklist.len());
        }

        // Extract all values from cache and create dependency list
        dependency_set = cache.into_values().collect();

        return dependency_set;
    }

    /// Computes a set of packages that need to be installed to satisfy package_name's deps given the current installed packages.
    /// When a dependency A | B | C is unsatisfied, there are two possible cases:
    ///   (1) there are no versions of A, B, or C installed; pick the alternative with the highest version number (yes, compare apples and oranges).
    ///   (2) at least one of A, B, or C is installed (say A, B), but with the wrong version; of the installed packages (A, B), pick the one with the highest version number.
    pub fn compute_how_to_install(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        
        #[allow(unused_variables, unused_assignments)]
        let mut dependencies_to_add : Vec<i32> = vec![];

        // implement more sophisticated worklist
        let mut deps : &Vec<Dependency> = &*self.dependencies.get(self.get_package_num(package_name)).unwrap();

        // Create hashmaps to cache and function as worklist
        let mut cache: HashMap<String, i32> = HashMap::new();
        let mut worklist: VecDeque<String> = VecDeque::new();

        // implement worklist
        // Insert initial list of dependancies to worklist
        self.compute_add_deps_to_worklist(&mut worklist, &cache, deps);
        // println!("=> Initial Worklist len: {}", worklist.len());

        // continue until worklist is empty
        let mut curr_dep_num: i32;
        while !worklist.is_empty() {
           
            // get value from the front of the queue
            match worklist.pop_front() {
                None => (),
                Some(val) => {
                    
                    // get package number
                    curr_dep_num = self.get_package_num(&val).clone();
                    cache.insert(val, curr_dep_num);

                    // Get deps
                    deps = &*self.dependencies.get(&curr_dep_num).unwrap();
                    self.compute_add_deps_to_worklist(&mut worklist, &cache, deps);
                }
            }

            // For debugging
            // println!("Worklist length: {}", worklist.len());
        }

        // Extract all values from cache and create dependency list
        dependencies_to_add = cache.into_values().collect();

        return dependencies_to_add;
    }

    // Helper functions

    fn transitive_add_deps_to_worklist(&self, worklist: &mut VecDeque<String>, cache: &HashMap<String, i32>, deps: &Vec<Dependency>) -> () {
       
        let mut dep_struct: &RelVersionedPackageNum; 
        let mut alt_num: i32; 
        let mut alt_name: &str; 
        
        // Iterate through dependency vec and all applicable ones to the worklist
        for dep in deps {
            
            // Get first alterate every time
            dep_struct = &dep[0]; 
            alt_num = dep_struct.package_num; 
            alt_name = self.get_package_name(alt_num);

            // verify its not in the cache 
            if cache.contains_key(alt_name) {
                continue;
            }

            // If not in cache save to worklist
            worklist.push_back(String::from(alt_name));
        }
    }
    
    fn compute_add_deps_to_worklist(&self, worklist: &mut VecDeque<String>, cache: &HashMap<String, i32>, deps: &Vec<Dependency>) -> () {
       
        let mut dep_struct: &RelVersionedPackageNum; 
        let mut alt_num: i32; 
        let mut alt_name: &str; 
        
        // Iterate through dependency vec and all applicable ones to the worklist
        for dep in deps {

            // check if installed first 
            if self.dep_is_satisfied(&dep).is_none() {
            
                // check if dep is satisfied already
                let somewhat_satisfied: Vec<&str> = self.dep_satisfied_by_wrong_version(&dep);

                // if length is 0, then its not installed at all
                if somewhat_satisfied.len() == 0 {
                    
                    // nothing is installed but we must check if multiple alternates are available
                    if dep.len() == 1 {

                        // Since length is one, retrieve first element
                        dep_struct = &dep[0]; 
                        alt_num = dep_struct.package_num; 
                        alt_name = self.get_package_name(alt_num);

                        // verify its not in the cache 
                        if cache.contains_key(alt_name) {
                            continue;
                        }

                        // If not in cache save to worklist
                        worklist.push_back(String::from(alt_name));
                    }

                    // there are mutliple alternates that are not installed
                    else {
                       
                        // find alternate with largest version number
                        // default to the first value
                        let mut highest_alt: &str = self.get_package_name(dep[0].package_num);
                        let (_op, base_ver) = dep[0].rel_version.as_ref().unwrap();
                        let mut base_ver1 = base_ver.parse::<debversion::DebianVersionNum>().unwrap(); 

                        let mut ver2;
                        let op: VersionRelation = VersionRelation::GreaterOrEqual;
                        for i in 1..(dep.len()) {

                            // get value at next index
                            let (_op, temp_ver) = dep[i].rel_version.as_ref().unwrap();
                            ver2 = temp_ver.parse::<debversion::DebianVersionNum>().unwrap(); 

                            // compare
                            // If next verion is higher than the default, save it as the highest
                            if debversion::cmp_debversion_with_op(&op, &ver2, &base_ver1) {
                                highest_alt = self.get_package_name(dep[i].package_num);                            
                                base_ver1 = ver2;
                            }
                        }
                        
                        // verify its not in the cache 
                        if cache.contains_key(highest_alt) {
                            continue;
                        }
                        
                        // If not in cache save to worklist
                        worklist.push_back(String::from(highest_alt));
                    } 
                } else {
                   
                    // Otherwise, there are multiple alternates installed with the wrong number 
                    // find alternate with largest version number
                    // default to the first value
                    let mut highest_alt: &str = somewhat_satisfied[0];
                    let mut base_ver1 = self.get_installed_debver(highest_alt).unwrap(); 

                    let mut ver2;
                    let op: VersionRelation = VersionRelation::GreaterOrEqual;
                    for i in 1..(somewhat_satisfied.len()) {

                        // get value at next index
                        ver2 = self.get_installed_debver(somewhat_satisfied[i]).unwrap(); 

                        // compare
                        // If next verion is higher than the default, save it as the highest
                        if debversion::cmp_debversion_with_op(&op, &ver2, &base_ver1) {
                            highest_alt = somewhat_satisfied[i];                            
                            base_ver1 = ver2;
                        }
                    }
                    
                    // verify its not in the cache 
                    if cache.contains_key(highest_alt) {
                        continue;
                    }

                    // If not in cache save to worklist
                    worklist.push_back(String::from(highest_alt));

                }
            }
        }
    }

}

