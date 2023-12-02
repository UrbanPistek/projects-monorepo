use super::{checksum::Checksum, idea::Idea, package::Package, Event};
use crossbeam::channel::{Receiver, Sender};

pub struct Student {
    idea: Option<Idea>,
    pkgs: Vec<Package>,
    skipped_idea: bool,
    event_sender: Sender<Event>,
    event_recv: Receiver<Event>,
}

impl Student {
    pub fn new(event_sender: Sender<Event>, event_recv: Receiver<Event>) -> Self {
        Self {
            event_sender,
            event_recv,
            idea: None,
            pkgs: vec![],
            skipped_idea: false,
        }
    }

    fn build_idea(
        &mut self,
        idea_checksum:&mut Checksum,
        pkg_checksum: &mut Checksum,
    ) {
        if let Some(ref idea) = self.idea {
            // Can only build ideas if we have acquired sufficient packages
            let pkgs_required = idea.num_pkg_required;
            if pkgs_required <= self.pkgs.len() {

                // Update idea and package checksums
                // All of the packages used in the update are deleted, along with the idea
                idea_checksum.update(Checksum::with_sha256(&idea.name));

                let pkgs_used = self.pkgs.drain(0..pkgs_required).collect::<Vec<_>>();
                for pkg in pkgs_used.iter() {
                    pkg_checksum.update(Checksum::with_sha256(&pkg.name));
                }

                self.idea = None;
            }
        }
    }

    pub fn run(&mut self, idea_checksum_copy: Checksum, pkg_checksum_copy: Checksum) -> (Checksum, Checksum){

        let mut idea_checksum = idea_checksum_copy;
        let mut pkg_checksum = pkg_checksum_copy;
        
        loop {
            
            let event = self.event_recv.recv().unwrap();
            match event {
                Event::NewIdea(idea) => {
                    // If the student is not working on an idea, then they will take the new idea
                    // and attempt to build it. Otherwise, the idea is skipped.
                    if self.idea.is_none() {
                        self.idea = Some(idea);
                        self.build_idea(&mut idea_checksum, &mut pkg_checksum);
                    } else {
                        self.event_sender.send(Event::NewIdea(idea)).unwrap();
                        self.skipped_idea = true;
                    }
                }

                Event::DownloadComplete(pkg) => {
                    // Getting a new package means the current idea may now be buildable, so the
                    // student attempts to build it
                    self.pkgs.push(pkg);
                    self.build_idea(&mut idea_checksum, &mut pkg_checksum);
                }

                Event::OutOfIdeas => {
                    // If an idea was skipped, it may still be in the event queue.
                    // If the student has an unfinished idea, they have to finish it, since they
                    // might be the last student remaining.
                    // In both these cases, we can't terminate, so the termination event is
                    // deferred ti the back of the queue.
                    if self.skipped_idea || self.idea.is_some() {
                        self.event_sender.send(Event::OutOfIdeas).unwrap();
                        self.skipped_idea = false;
                    } else {
                        // Any unused packages are returned to the queue upon termination
                        for pkg in self.pkgs.drain(..) {
                            self.event_sender
                                .send(Event::DownloadComplete(pkg))
                                .unwrap();
                        }
                        return (idea_checksum, pkg_checksum);
                    }
                }
            }
        }
    }
}
