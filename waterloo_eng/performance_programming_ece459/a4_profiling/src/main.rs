#![warn(clippy::all)]
use lab4::{
    checksum::Checksum, idea::IdeaGenerator, package::PackageDownloader, student::Student, Event, get_packages, generate_ideas_vec,
};
use crossbeam::channel::{bounded, Receiver, Sender};
use std::env;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread::spawn;

struct Args {
    pub num_ideas: usize,
    pub num_idea_gen: usize,
    pub num_pkgs: usize,
    pub num_pkg_gen: usize,
    pub num_students: usize,
    pub customer_ideas_file: String,
    pub product_ideas_file: String,
    pub packages_file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect(); 
    let num_ideas = args.get(1).map_or(Ok(80), |a| a.parse())?;
    let num_idea_gen = args.get(2).map_or(Ok(2), |a| a.parse())?;
    let num_pkgs = args.get(3).map_or(Ok(4000), |a| a.parse())?;
    let num_pkg_gen = args.get(4).map_or(Ok(6), |a| a.parse())?;
    let num_students = args.get(5).map_or(Ok(6), |a| a.parse())?;
    
    let customer_ideas_file: String = args.get(6).map(|a| a.clone()).unwrap_or("data/ideas-customers.txt".to_string());
    let product_ideas_file: String = args.get(7).map(|a| a.clone()).unwrap_or("data/ideas-products.txt".to_string());
    let packages_file: String = args.get(8).map(|a| a.clone()).unwrap_or("data/packages.txt".to_string());

    let args = Args {
        num_ideas,
        num_idea_gen,
        num_pkgs,
        num_pkg_gen,
        num_students,
        customer_ideas_file,
        product_ideas_file,
        packages_file,
    };

    hackathon(&args);
    Ok(())
}

fn per_thread_amount(thread_idx: usize, total: usize, threads: usize) -> usize {
    let per_thread = total / threads;
    let extras = total % threads;
    per_thread + (thread_idx < extras) as usize
}

fn hackathon(args: &Args) {

    // Read in all I/O first
    let pkgs = get_packages(args.num_pkgs, &args.packages_file);
    let ideas_vec = generate_ideas_vec(&args.product_ideas_file, &args.customer_ideas_file);

    // Use message-passing channel as event queue
    let (send, recv) = bounded::<Event>(args.num_pkgs);

    let mut threads = vec![];
    let mut download_threads = vec![];
    let mut student_threads = vec![];

    // Checksums of all the generated ideas and packages
    let mut idea_checksum = Arc::new(Mutex::new(Checksum::default()));

    // Spawn all sets of threads concurrently so that they can start doing work while they continue
    // to spawn 
    let mut num_student_threads = 0; 
    let mut num_download_threads = 0; 
    let mut num_idea_threads = 0; 

    let mut dwn_start_idx = 0;
    let mut idea_start_idx = 0;
    
    loop {

        // Spawn student threads
        if num_student_threads < args.num_students {
            let mut student = Student::new(Sender::clone(&send), Receiver::clone(&recv));
            let thread = spawn(move || student.run(Checksum::default(), Checksum::default()));
            student_threads.push(thread);

            // increment 
            num_student_threads += 1;
        }

        // Spawn package downloader threads. Packages are distributed evenly across threads.
        if num_download_threads < args.num_pkg_gen {

            // determine number of packages to allocate to each thread
            let num_pkgs = per_thread_amount(num_download_threads, args.num_pkgs, args.num_pkg_gen);

            // copy specific slice of packages to download to each thread
            let pkg_slice = pkgs[dwn_start_idx..(dwn_start_idx + num_pkgs)].to_owned();
            
            let downloader = PackageDownloader::new(num_pkgs, Sender::clone(&send), pkg_slice);
            dwn_start_idx += num_pkgs;

            let thread = spawn(move || downloader.run(Checksum::default()));
            download_threads.push(thread);

            // increment
            num_download_threads += 1;
        }

        // Spawn idea generator threads. Ideas and packages are distributed evenly across threads. In
        // each thread, packages are distributed evenly across ideas.
        if num_idea_threads < args.num_idea_gen {
            let num_ideas = per_thread_amount(num_idea_threads, args.num_ideas, args.num_idea_gen);
            let num_pkgs = per_thread_amount(num_idea_threads, args.num_pkgs, args.num_idea_gen);
            let num_students = per_thread_amount(num_idea_threads, args.num_students, args.num_idea_gen);
            let generator = IdeaGenerator::new(
                idea_start_idx,
                num_ideas,
                num_students,
                num_pkgs,
                Sender::clone(&send),
                ideas_vec.clone(),
            );
            let idea_checksum = Arc::clone(&idea_checksum);
            idea_start_idx += num_ideas;

            let thread = spawn(move || generator.run(idea_checksum));
            threads.push(thread);

            // increment
            num_idea_threads += 1;
        }

        // Break out of the loop
        if (num_student_threads >= args.num_students) && (num_download_threads >= args.num_pkg_gen) && (num_idea_threads >= args.num_idea_gen) {
            break;
        }

    }

    assert_eq!(dwn_start_idx, args.num_pkgs);
    assert_eq!(idea_start_idx, args.num_ideas);
    
    // Join all threads
    threads.into_iter().for_each(|t| t.join().unwrap());

    // Join all download threads 
    let pkg_checksum_res = download_threads.into_iter().map(|handle| {
        handle.join().unwrap()
    }).collect::<Vec<_>>();
    
    let mut pkg = Checksum::default();
    for check in pkg_checksum_res {
        pkg.update(check);
    }

    // Join all student threads
    let student_threads_res = student_threads.into_iter().map(|handle| {
        handle.join().unwrap()
    }).collect::<Vec<_>>();

    let mut student_idea = Checksum::default();
    let mut student_pkg = Checksum::default();
    for check in student_threads_res {
        student_idea.update(check.0);
        student_pkg.update(check.1);
    }

    let idea = Arc::get_mut(&mut idea_checksum).unwrap().get_mut().unwrap();

    println!("Global checksums:\nIdea Generator: {}\nStudent Idea: {}\nPackage Downloader: {}\nStudent Package: {}", 
        idea, student_idea, pkg, student_pkg);
}
