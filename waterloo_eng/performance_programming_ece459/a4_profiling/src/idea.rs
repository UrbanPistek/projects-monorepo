use super::checksum::Checksum;
use super::Event;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

pub struct Idea {
    pub name: String,
    pub num_pkg_required: usize,
}

pub struct IdeaGenerator {
    idea_start_idx: usize,
    num_ideas: usize,
    num_students: usize,
    num_pkgs: usize,
    event_sender: Sender<Event>,
    ideas: Vec<(String, String)>,
    ideas_len: usize,
}

impl IdeaGenerator {
    pub fn new(
        idea_start_idx: usize,
        num_ideas: usize,
        num_students: usize,
        num_pkgs: usize,
        event_sender: Sender<Event>,
        ideas: Vec<(String, String)>,
    ) -> Self {
        let ideas_len = ideas.len();
        Self {
            idea_start_idx,
            num_ideas,
            num_students,
            num_pkgs,
            event_sender,
            ideas,
            ideas_len,
        }
    }

    pub fn run(&self, idea_checksum: Arc<Mutex<Checksum>>) {
        let pkg_per_idea = self.num_pkgs / self.num_ideas;
        let extra_pkgs = self.num_pkgs % self.num_ideas;

        // Generate a set of new ideas and place them into the event-queue
        // Update the idea checksum with all generated idea names
        let mut idx; 
        let mut pair;
        let mut name; 
        let mut extra;
        let mut num_pkg_required;
        for i in 0..self.num_ideas {

            idx = self.idea_start_idx + i;
            pair = self.ideas[idx % self.ideas_len].to_owned();
            name = format!("{} for {}", pair.0, pair.1);

            extra = (i < extra_pkgs) as usize;
            num_pkg_required = pkg_per_idea + extra;
            let idea = Idea {
                name,
                num_pkg_required,
            };

            idea_checksum
                .lock()
                .unwrap()
                .update(Checksum::with_sha256(&idea.name));

            self.event_sender.send(Event::NewIdea(idea)).unwrap();

        }

        // Push student termination events into the event queue
        for _ in 0..self.num_students {
            self.event_sender.send(Event::OutOfIdeas).unwrap();
        }
    }
}
