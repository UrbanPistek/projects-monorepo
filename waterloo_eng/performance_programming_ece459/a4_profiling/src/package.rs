use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;

pub struct Package {
    pub name: String,
}

pub struct PackageDownloader {
    num_pkgs: usize,
    event_sender: Sender<Event>,
    pkg_slice: Vec<String>,
}

impl PackageDownloader {
    pub fn new(num_pkgs: usize, event_sender: Sender<Event>, pkg_slice: Vec<String>) -> Self {
        
        Self {
            num_pkgs,
            event_sender,
            pkg_slice, 
        }
    }

    pub fn run(&self, pkg_checksum_copy: Checksum) -> Checksum{
        
        let mut name;
        
        let mut cs = pkg_checksum_copy;

        // Generate a set of packages and place them into the event queue
        // Update the package checksum with each package name
        for i in 0..self.num_pkgs {

            name = self.pkg_slice[i].to_owned();
            cs.update(Checksum::with_sha256(&name));

            self.event_sender
                .send(Event::DownloadComplete(Package { name }))
                .unwrap();
        }

        return cs
    }
}
