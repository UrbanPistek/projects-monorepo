use rpkg::debversion;
use crate::Packages;
use crate::packages::Dependency;

impl Packages {
    /// Gets the dependencies of package_name, and prints out whether they are satisfied (and by which library/version) or not.
    pub fn deps_available(&self, package_name: &str) {
        if !self.package_exists(package_name) {
            println!("no such package {}", package_name);
            return;
        }
        println!("Package {}:", package_name);

        // Get package number
        let current_package_num: &i32 = self.get_package_num(package_name);
        let deps : &Vec<Dependency> = &*self.dependencies.get(&current_package_num).unwrap();

        // Iterate over all dependancies for a package 
        // For each, check if a package is installed in the installed hash map
        // Check to see if the version  number is compatible 
        for dep in deps {
            println!("- dependency {:?}", self.dep2str(&dep));

            // Verify if dependacy is already installed
            match self.dep_is_satisfied(&dep) {
                None => {
                    println!("-> not satisfied");
                },
                Some(installed_package_name) => {

                    // get the installed version number
                    let ver = self.get_installed_debver(&installed_package_name).unwrap();

                    println!("+ {} satisfied by installed version {}", installed_package_name, ver);
                }
            }
        }
    }

    /// Returns Some(package) which satisfies dependency dd, or None if not satisfied.
    pub fn dep_is_satisfied(&self, dd:&Dependency) -> Option<&str> {

        // loop through the alternates and check installed to determine if the dependency is installed
        for alt in dd {
            
            // unpack alt
            // get num and then name from num
            let alt_num: i32 = alt.package_num; 
            let alt_name: &str = self.get_package_name(alt_num);

            // Unpack relation and version number
            let (op, ver_unpacked) = &alt.rel_version.as_ref().unwrap();
            let ver = ver_unpacked.parse::<debversion::DebianVersionNum>().unwrap(); 
            // println!("Upacked: {} with op: {} and version: {}", alt_name, op, ver_unpacked);
            
            // Unpackage result to determine if package is installed
            match self.get_installed_debver(alt_name) {
                None => continue, 
                Some(installed_ver) => {
                    // println!("Dependancy is already installed: {} with version: {}", alt_name, installed_ver); 

                    // Check installed version with available to see if relation holds
                    let satisfied = debversion::cmp_debversion_with_op(&op, &installed_ver, &ver);
                    // println!("Satisfied: {}, ", satisfied);
                
                    // return if satisfied 
                    if satisfied {
                        return Some(&alt_name);
                    }

                }
            }

        }

        return None;
    }

    /// Returns a Vec of packages which would satisfy dependency dd but for the version.
    /// Used by the how-to-install command, which calls compute_how_to_install().
    pub fn dep_satisfied_by_wrong_version(&self, dd:&Dependency) -> Vec<&str> {
        assert! (self.dep_is_satisfied(dd).is_none());
        let mut result = vec![];

        // loop through the alternates and check installed to determine if the dependency is installed
        for alt in dd {
            
            // unpack alt
            // get num and then name from num
            let alt_num: i32 = alt.package_num; 
            let alt_name: &str = self.get_package_name(alt_num);
            
            // Unpackage result to determine if package is installed
            match self.get_installed_debver(alt_name) {
                None => continue, 
                Some(_v) => {
                
                    // Since deps_is_satisfied already ran if the package is installed
                    // We know its not satisfied because of a wrong version number
                    result.push(alt_name);
                }
            }

        }
        
        return result;

    }
}

