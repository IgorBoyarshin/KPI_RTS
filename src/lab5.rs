use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;

extern crate rand;
extern crate cairo;
extern crate num;

fn erlang_next(rng: &mut SmallRng, k: u32, lambda: f64) -> f64 {
    let mut acc = 0.0;
    for _ in 0..k {
        acc += -1.0 / lambda * (rng.gen::<f64>()).ln();
    }
    acc / k as f64
}

fn erlang_for_mean(mean: f64, rng: &mut SmallRng) -> f64 {
    let k = 16;
    erlang_next(rng, k, 1.0 / mean)
}

struct Spawn {
    time: f64,
    task_type: TaskType,
}

struct System {
    spawns: Vec<Spawn>, // end is top
    tasks: Vec<SystemTask>, // end is top
    processed: u32,
    deadlines_missed: u32,
}

impl System {
    fn new() -> System {
        let mut system = System { spawns: Vec::new(), tasks: Vec::new(),
            deadlines_missed: 0, processed: 0,};
        system.add_spawn(Spawn{time: 0.1, task_type: TaskType::Long});
        system.add_spawn(Spawn{time: 0.2, task_type: TaskType::Medium});
        system.add_spawn(Spawn{time: 0.3, task_type: TaskType::Small});
        system
    }

    fn add_spawn(&mut self, new_spawn: Spawn) {
        let mut index = self.spawns.len();
        while index > 0 && (self.spawns[index - 1].time < new_spawn.time) { index -= 1 };
        self.spawns.insert(index, new_spawn);
    }

    fn add_system_task(&mut self, new_task: SystemTask) {
        let mut index = self.tasks.len();
        while index > 0 && (new_task.more_important_than(&self.tasks[index - 1])) { index -= 1 };
        self.tasks.insert(index, new_task);
    }

    fn simulate(&mut self, simulation_time: f64, rng: &mut SmallRng) {
        println!("========== Simulation start ({}) ==========", simulation_time);
        const DEADLINE_MULTIPLIER: f64 = 4.0;
        let mut current_time = 0.0;
        let mut count = 0;
        let mut stats = Vec::new();
        let mut free_time = 0.0;
        while current_time < simulation_time {
            count += 1;
            // println!("------Starting iteration at time {:.2}", current_time);
            let earliest_finish =
                if let Some(&SystemTask {left_to_run, ..}) = self.tasks.last() {
                    current_time + left_to_run
                } else { std::f64::INFINITY };
            let earliest_spawn = self.spawns.last().unwrap().time;

            if earliest_spawn < earliest_finish {
                let time_delta = earliest_spawn - current_time;
                current_time = earliest_spawn;
                if self.tasks.is_empty() {
                    free_time += time_delta;
                }
                // println!("--------Fast forward to {:.2}", current_time);
                // println!("Earliest was spawn");

                // Do the spawn and prepare another one in place of it
                let Spawn {task_type, ..} = self.spawns.pop().unwrap();
                let Task {exec, period} = task_from_type(task_type, rng);
                self.add_spawn(Spawn {
                    time: current_time + period,
                    task_type
                });
                // println!("Generating spawn with time={:.2} and type={:?}", current_time + period, task_type);
                // Advance progress on current task
                if let Some(task) = self.tasks.last_mut() {
                    task.left_to_run -= time_delta;
                    // println!("Advancing current task from {:.2} to {:.2}",
                        // task.left_to_run + time_delta, task.left_to_run);
                }
                // Insert the new task
                self.add_system_task(SystemTask {
                    left_to_run: exec,
                    execution_time: exec,
                    deadline: current_time + DEADLINE_MULTIPLIER * exec,
                    task_type: task_type,
                });
                // println!("Adding task with time={:.2} and deadline={:.2} and type={:?}",
                    // exec, current_time + DEADLINE_MULTIPLIER * exec, task_type);
                // let curr = &self.tasks.last().unwrap();
                // println!("$$$$ Now current is: left={}, deadline={}, type={:?}",
                    // curr.left_to_run, curr.deadline, curr.task_type);
            } else { // finish first
                // let time_delta = earliest_finish - current_time;
                current_time = earliest_finish;
                // println!("-------Fast forward to {:.2}", current_time);
                // println!("Earliest was task finish");

                // There are sure to be at least 1 tasks because earliest_finish
                // wouldn't have been first otherwise.
                // We are sure that the current task has finished because we've
                // jumped onto its finish time.
                let SystemTask{deadline, execution_time, ..} = self.tasks.pop().unwrap();
                if !self.tasks.is_empty() {
                    // let curr = &self.tasks.last().unwrap();
                    // println!("$$$$ Now current is: left={}, deadline={}, type={:?}",
                    //     curr.left_to_run, curr.deadline, curr.task_type);
                }
                self.processed += 1;
                if current_time > deadline {
                    // println!("Deadline was not met ({:.2} > {:.2})", current_time, deadline);
                    // expired
                    self.deadlines_missed += 1;
                    stats.push(0.0);
                } else { // all good
                    // all good
                    // println!("Deadline was met ({:.2} < {:.2})", current_time, deadline);
                    stats.push((deadline - current_time) / (DEADLINE_MULTIPLIER * execution_time));
                }
            }
        }
        println!("========== Simulation end ==========");
        println!("stats:");
        println!("System load = {}", 1.0 - free_time / current_time);
        println!("{} iterations done", count);
        println!("{} tasks processed", self.processed);
        println!("{} tasks left in the system", self.tasks.len());
        println!("----- Accounting for leftovers...");
        // println!("({} additional deadlines missed)", self.tasks.iter()
        //     .filter(|task| task.deadline < current_time).count());
        println!("{} deadlines missed in total", self.deadlines_missed +
                 self.tasks.iter().filter(|task| task.deadline < current_time).count() as u32);
        let mut mean = 0.0;
        let mut stats_len = stats.len();
        for ratio in stats {
            mean += ratio;
        }
        for task in self.tasks.iter() {
            if task.deadline < current_time {
                stats_len += 1;
            }
        }
        mean /= stats_len as f64;
        println!("Avg space utilization = {}", 1.0 - mean);
    }
}

struct SystemTask {
    left_to_run: f64,
    execution_time: f64,
    deadline: f64,
    task_type: TaskType,
}

struct Task {
    exec: f64,
    period: f64,
}

#[derive(Copy, Clone, Debug)]
enum TaskType {
    Small,
    Medium,
    Long,
}

impl SystemTask {
    fn more_important_than(&self, other: &SystemTask) -> bool {
        self.rang_priority() > other.rang_priority()
    }

    fn rang_priority(&self) -> u32 {
        match self.task_type {
            TaskType::Small  => 0,
            TaskType::Medium => 1,
            TaskType::Long   => 2,
        }
    }
}

fn task_from_type(task_type: TaskType, rng: &mut SmallRng) -> Task {
    const LAMBDA: f64 = 0.1;
    let (exec, period) = match task_type {
        TaskType::Small  => (6.0, 6.0 / LAMBDA),
        TaskType::Medium => (12.0, 12.0 / LAMBDA),
        TaskType::Long   => (20.0, 20.0 / LAMBDA),
    };
    let exec = erlang_for_mean(exec, rng);
    let period = erlang_for_mean(period, rng);

    Task {exec, period}
}

pub fn lab5_work() {
    let seed = [10,4,3,8, 7,9,8,10, 14,18,12,12, 14,15,16,17];
    let mut rng = SmallRng::from_seed(seed);
    let mut system = System::new();
    system.simulate(10000.0, &mut rng);
}

// use cairo::{ImageSurface, Format, Context};
// use std::fs::File;
// use std::time::{Instant};
// use num::complex::Complex;
// use rand::distributions::{Poisson, Distribution};



// fn dispersion(v: &Vec<f64>, mean: f64) -> f64 {
//     v.iter().map(|x| (x - mean)*(x - mean)).sum::<f64>() / ((v.len() - 1) as f64)
// }

// fn mean(v: &Vec<f64>) -> f64 {
//     v.iter().sum::<f64>() / v.len() as f64
// }
//
// fn correlation(v1: &Vec<f64>, mean1: f64, v2: &Vec<f64>, mean2: f64) -> Vec<f64> {
//     (0..(v1.len() / 2))
//         .map(|tau| (v1.iter().zip(v2.iter().skip(tau)), v1.len() - tau))
//         .map(|(v, _size)| (v.map(|(a, b)| (a - mean1) * (b - mean2)), _size))
//         .map(|(v, size)| v.sum::<f64>() / (size - 1) as f64)
//         .collect::<Vec<f64>>()
// }

// fn fast_fourier(signal: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
//     if signal.len() == 2 {
//         return vec![signal[0] + signal[1], signal[0] - signal[1]]
//     }
//
//     let even = fast_fourier(&signal.iter().step_by(2).map(|x| *x).collect());
//     let odd = fast_fourier(&signal.iter().skip(1).step_by(2).map(|x| *x).collect());
//     #[allow(non_snake_case)]
//     let N = signal.len();
//     let w = |k| {
//         if k % N == 0 { Complex::new(1.0, 0.0) }
//         else {
//             let arg = -2.0 * std::f64::consts::PI / N as f64 * k as f64;
//             Complex::new(arg.cos(), arg.sin())
//         }
//     };
//     let mut result = Vec::new();
//     result.append(&mut even.iter().zip(odd.iter()).enumerate()
//         .map(|(i, (e, o))| e + w(i) * o).collect::<Vec<_>>());
//     result.append(&mut even.iter().zip(odd.iter()).enumerate()
//         .map(|(i, (e, o))| e - w(i) * o).collect::<Vec<_>>());
//     result
// }
//
// fn discrete_fourier(signal: &Vec<f64>) -> Vec<f64> {
//     #[allow(non_snake_case)]
//     let N = signal.len();
//     // (0..N / 2).map(|p| {
//     (0..N).map(|p| {
//         let (re, im) = signal.iter().enumerate()
//             .map(|(k, x)| {
//                 let arg = 2.0 * std::f64::consts::PI / N as f64 * p as f64 * k as f64;
//                 (x * arg.cos(), x * arg.sin())
//             })
//             .fold((0.0, 0.0), |(re, im), (next_re, next_im)| (re + next_re, im + next_im));
//         (re*re + im*im).sqrt()
//     })
//     // .map(|f| f * 2.0 / N as f64)
//     .collect()
// }
//
// fn func(amplitude: f64, frequency: f64, phase: f64, time: f64) -> f64 {
//     amplitude * (frequency * time + phase).sin()
// }
//
// fn construct_signal(rng: &mut SmallRng,
//                     harmonics_count: usize,
//                     marginal_frequency: usize,
//                     timespan: usize) -> Vec<f64> {
//     const MAX_AMPLITUDE: f64 = 1.0;
//     const MAX_PHASE:     f64 = 2.0 * std::f64::consts::PI;
//     let frequency_step = marginal_frequency as f64 / harmonics_count as f64;
//     std::iter::repeat(frequency_step).take(harmonics_count).scan(0.0, |acc, curr| {*acc += curr; Some(*acc)})
//         .map(|frequency| {
//             let frequency = frequency as f64;
//             let amplitude = rng.gen::<f64>() * MAX_AMPLITUDE;
//             let phase     = rng.gen::<f64>() * MAX_PHASE;
//             (frequency, amplitude, phase)
//         })
//         .map(|(f, a, p)| (0..timespan).map(|t| func(a, f, p, t as f64)).collect::<Vec<_>>())
//         .fold(vec![0.0; timespan], |sum, harmonic|
//             sum.into_iter()
//             .zip(harmonic.into_iter())
//             .map(|(a, b)| a + b)
//             .collect())
// }
//
//
// struct Renderer {
//     surface: ImageSurface,
//     context: Context,
//     width: i32,
//     height: i32,
// }
//
// impl Renderer {
//     fn with_width_height(width: i32, height: i32) -> Renderer {
//         let surface = ImageSurface::create(Format::ARgb32, width, height).unwrap();
//         let context = Context::new(&surface);
//         context.set_font_size(20.0);
//         Renderer {
//             surface,
//             context,
//             width,
//             height
//         }
//     }
//
//     fn export_to(&self, filename: &str) {
//         let mut file = File::create(filename).expect("Can't create file");
//         self.surface.write_to_png(&mut file).expect("Can't write to png");
//     }
//
//     fn clear(&self) {
//         self.context.set_source_rgb(1.0, 1.0, 1.0);
//         self.context.paint();
//     }
//
//     fn draw_func(&self, ys: &Vec<f64>, xs: Option<Vec<usize>>) {
//         let (w, h) = (self.width as f64, self.height as f64);
//         let (origin_x, origin_y) = (5.0, h / 2.0);
//         let max_y = ys.iter().fold(std::f64::NEG_INFINITY, |a, &b| a.max(b));
//         let scale_y = (h / 2.0) / max_y;
//         let step_x = (w - origin_x) / ys.len() as f64;
//
//         self.context.set_source_rgb(1.0, 0.0, 0.0);
//         self.context.set_line_width(2.0);
//         let mut x = origin_x;
//         self.context.move_to(x, origin_y - ys[0] * scale_y);
//         x += step_x;
//         for y in ys.iter().skip(1) {
//             let y = origin_y - y * scale_y; // y axis is inverted
//             self.context.line_to(x, y);
//             x += step_x;
//         }
//         self.context.stroke();
//
//         // Draw Y axis hints
//         self.context.set_source_rgb(0.0, 0.0, 1.0);
//         let y_amount = 4 as f64;
//         let step_y = h / 2.0 / (y_amount + 1.0);
//         let y_step = max_y / (y_amount + 1.0);
//         let mut y = y_step * y_amount;
//         for i in 1..=4 {
//             self.context.move_to(7.0, i as f64 * step_y);
//             let mut text = y.to_string();
//             text.truncate(5);
//             self.context.show_text(&text);
//             y -= y_step;
//         }
//         self.context.stroke();
//
//         // Maybe draw X axis hints (expects them to be uniformly distributed)
//         if let Some(xs) = xs {
//             const AMOUNT: usize = 5;
//             let chunk_size = xs.len() / AMOUNT;
//             self.context.set_source_rgb(0.0, 0.0, 1.0);
//             for (index, x) in xs.into_iter().enumerate().step_by(chunk_size) {
//                 self.context.move_to(5.0 + index as f64 * step_x, origin_y - 5.0);
//                 self.context.show_text(&x.to_string());
//             }
//             self.context.stroke();
//         }
//     }
//
//     fn draw_axis(&self) {
//         let (w, h) = (self.width as f64, self.height as f64);
//         let (origin_x, origin_y) = (5.0, h / 2.0);
//         self.context.set_source_rgb(0.0, 0.0, 0.0);
//         self.context.set_line_width(4.0);
//
//         // Y axis
//         self.context.move_to(origin_x, 0.0);
//         self.context.line_to(origin_x, h);
//         // X axis
//         self.context.move_to(0.0, origin_y);
//         self.context.line_to(w, origin_y);
//
//         self.context.stroke();
//     }
// }






// fn factorial(n: u32) -> u32 {
//     let mut acc = 1;
//     for curr in 1..=n {
//         acc *= curr;
//     }
//     acc
// }


// struct Erlang {
//     k: u32,
//     lambda: f64,
// }

// impl Erlang {
//     // fn new(k: u32, lambda: f64) -> Erlang {
//     //     Erlang {k, lambda}
//     // }
//
//     // fn next(&self, rng: &mut SmallRng) -> f64 {
//     //     let mut acc = 0.0;
//     //     for _ in 0..self.k {
//     //         acc += -1.0/self.lambda * (rng.gen::<f64>()).ln();
//     //     }
//     //     acc / self.k as f64
//     // }
//
//     fn next(rng: &mut SmallRng, k: u32, lambda: f64) -> f64 {
//         let mut acc = 0.0;
//         for _ in 0..k {
//             acc += -1.0 / lambda * (rng.gen::<f64>()).ln();
//         }
//         acc / k as f64
//     }
//
//     fn gen_for_mean(mean: f64, rng: &mut SmallRng) -> f64 {
//         let k = 4;
//         Erlang::next(&mut rng, k, 1.0 / mean)
//     }
//
//     // fn next(&self, x: f64) -> f64 {
//     //     let k = self.k;
//     //     let lambda = self.lambda;
//     //     let e = std::f64::consts::E.powf(-lambda*x);
//     //     lambda.powi(k as i32) * x.powi(k as i32 - 1) * e / factorial(k - 1) as f64
//     // }
// }






    // const TIMESPAN_STEP: usize = 32;
    // let mut times = Vec::new();
    // // let timepoints: Vec<_> = (TIMESPAN_STEP..).step_by(TIMESPAN_STEP)
    // //     .take_while(|&span| span <= TIMESPAN).collect();
    // let timepoints: Vec<_> = std::iter::repeat(2)
    //     .scan(32, |acc, curr| {*acc *= curr; Some(*acc)})
    //     .take_while(|&x| x <= TIMESPAN)
    //     .collect();
    //
    // // for timespan in timepoints.iter() {
    // //     let signal_x = construct_signal(&mut rng, HARMONICS_COUNT, MARGINAL_FREQUENCY, *timespan);
    // //     let begin = Instant::now();
    // //         let _fast_fourier_x = fast_fourier(&signal_x.iter().map(|x| Complex::new(*x, 0.0)).collect());
    // //     let elapsed = begin.elapsed().as_micros();
    // //     times.push(elapsed as f64);
    // //     // println!("Elapsed for timespan={} : {}micros", timespan, elapsed);
    // // }
    //
    // let signal_x = construct_signal(&mut rng, HARMONICS_COUNT, MARGINAL_FREQUENCY, TIMESPAN);
    //     let begin = Instant::now();
    // let discrete_fourier_x = discrete_fourier(&signal_x);
    //     let elapsed = begin.elapsed().as_micros();
    //     println!("Elapsed for discrete: {}micros", elapsed);
    //     let begin = Instant::now();
    // let fast_fourier_x = fast_fourier(&signal_x.iter().map(|x| Complex::new(*x, 0.0)).collect())
    //     .into_iter().map(|c| c.norm_sqr().sqrt()).collect();
    //     let elapsed = begin.elapsed().as_micros();
    //     println!("Elapsed for fast: {}micros", elapsed);

    // let renderer = Renderer::with_width_height(1100, 600);
    // {
    //     renderer.clear();
    //     renderer.draw_axis();
    //     renderer.draw_func(&signal_x, None);
    //     renderer.export_to("signal_x.png");
    // }
    // {
    //     renderer.clear();
    //     renderer.draw_axis();
    //     renderer.draw_func(&discrete_fourier_x, None);
    //     renderer.export_to("discrete_fourier.png");
    // }
    // {
    //     renderer.clear();
    //     renderer.draw_axis();
    //     renderer.draw_func(&fast_fourier_x, None);
    //     renderer.export_to("fast_fourier.png");
    // }
    // {
    //     renderer.clear();
    //     renderer.draw_axis();
    //     renderer.draw_func(&times, Some(timepoints));
    //     renderer.export_to("times.png");
    // }
