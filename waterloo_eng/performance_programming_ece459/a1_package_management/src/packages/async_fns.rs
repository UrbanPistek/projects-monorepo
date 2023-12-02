use curl::easy::{Easy2, Handler, WriteError};
use curl::multi::{Easy2Handle, Multi};
use std::time::Duration;
use std::str;
use curl::Error;

use crate::Packages;

struct Collector(Box<String>);
impl Handler for Collector {
    fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
        (*self.0).push_str(str::from_utf8(&data.to_vec()).unwrap());
        Ok(data.len())
    }
}

const DEFAULT_SERVER : &str = "ece459.patricklam.ca:4590";
impl Drop for Packages {
    fn drop(&mut self) {
        self.execute()
    }
}

pub struct AsyncState {
    server : String,
    multi_handler : Multi,
    request_handler : Vec<Easy2Handle<Collector>>
}

impl AsyncState {
    pub fn new() -> AsyncState {
        AsyncState {
            server : String::from(DEFAULT_SERVER),
            multi_handler : Multi::new(),
            request_handler: Vec::new(),
        }
    }
}

impl Packages {
    pub fn set_server(&mut self, new_server:&str) {
        self.async_state.server = String::from(new_server);
    }

    /// Retrieves the version number of pkg and calls enq_verify_with_version with that version number.
    pub fn enq_verify(&mut self, pkg:&str) {
        let version = self.get_available_debver(pkg);
        match version {
            None => { println!("Error: package {} not defined.", pkg); return },
            Some(v) => { 
                let vs = &v.to_string();
                self.enq_verify_with_version(pkg, vs); 
            }
        };
    }

    /// Enqueues a request for the provided version/package information. Stores any needed state to async_state so that execute() can handle the results and print out needed output.
    pub fn enq_verify_with_version(&mut self, pkg:&str, version:&str) {
        let url = format!("http://{}/rest/v1/checksums/{}/{}", self.async_state.server, pkg, version);
        println!("queueing request {}", url);

        // Store handler in vec
        self.async_state.request_handler.push(self.handler_setup(&url).unwrap());

        // Get number of currently on-going requests - for debugging purposes
        // let num_request: u32 = self.async_state.multi_handler.perform().unwrap();
        // println!("Currently number of requests {}", num_request);

    }

    /// Asks curl to perform all enqueued requests. For requests that succeed with response code 200, compares received MD5sum with local MD5sum (perhaps stored earlier). For requests that fail with 400+, prints error message.
    pub fn execute(&mut self) {

        // block until all requests completed
        let mut num_request: u32 = self.async_state.multi_handler.perform().unwrap();
        while num_request > 0 {
            
            // Timeout
            self.async_state.multi_handler.wait(&mut [], Duration::from_secs(5)).unwrap();
            num_request = self.async_state.multi_handler.perform().unwrap();
            
            // print number of on-going requests for debugging
            // println!("Waiting on {} requests", num_request);
        }

        // unpack all handles
        for handle in self.async_state.request_handler.drain(..){
            
            // unpack handle
            let mut unpacked_handle = self.async_state.multi_handler.remove2(handle).unwrap();

            // Variables to store unpacked values
            let response_code: u32;
            let parsed_url: Vec<&str>;

            #[allow(unused_variables, unused_assignments)]
            let mut pkg_name: &str = "";

            #[allow(unused_variables, unused_assignments)]
            let mut version: &str = "";

            let pkg_num: &i32;
            let mut md5sum: &String = &String::from("");
            let data: String;
            let unpacked_url: Result<Option<&str>, Error>;

            // get data from response
            data = unpacked_handle.get_ref().0.to_string();

            // get response code
            response_code = unpacked_handle.response_code().unwrap();
            
            // unpack url from request to use to get the verion and package name 
            unpacked_url = unpacked_handle.effective_url();
          
            // using nested match statements to ensure values are matched
            // only if the previous was not None
            match unpacked_url.unwrap() {
                None => (),
                Some(url) => {
                    
                    parsed_url = url.split('/').collect();
                    version = parsed_url[parsed_url.len()-1];
                    pkg_name = parsed_url[parsed_url.len()-2];

                    match self.package_name_to_num.get(pkg_name) {
                        None => {
                            println!("Error: Package name not matched to number");
                        },
                        Some(num) => {
                            pkg_num = num;

                            match self.md5sums.get(pkg_num) {
                                None => {
                                    println!("Error: MD5sum not matched to package number");
                                },
                                Some(sum) => {
                                    md5sum = sum;
                                }
                            } 
                        }
                    } 

                    // Is considered successful if the response code is 200 
                    // and the md5sum is not empty
                    if response_code == 200 && md5sum != "" {
                        println!("verifying {}, matches: {:?}", pkg_name, data.eq(md5sum));
                    } else {
                        println!("got error {} on request for package {} version {}", response_code, pkg_name, version);
                    }
                }
            }
        }

    }

    fn handler_setup(&self, url: &str) -> Result<Easy2Handle<Collector>, Error> {
        
        // Create handle and add it to the multi handle
        let mut handle = Easy2::new(Collector(Box::new(String::new())));
        handle.url(url).unwrap();

        return Ok(self.async_state.multi_handler.add2(handle).unwrap())
    }

}
