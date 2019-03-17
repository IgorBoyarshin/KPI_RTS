use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;

use cairo::{ImageSurface, Format, Context};
use std::fs::File;
// use std::time::{Instant};

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

#[derive(Debug)]
enum SystemType {
    RateMonotonic,
    EarliestDeadlineFirst,
}

struct Stats {
    deadlines_missed: u32,
    system_load: f64,
    average_waiting_time: f64,
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

struct SystemTask {
    left_to_run: f64,
    execution_time: f64,
    deadline: f64,
    task_type: TaskType,
    birth_time: f64,
}

impl SystemTask {
    fn more_important_than(&self, other: &SystemTask, cmp_by_deadline: bool) -> bool {
        if cmp_by_deadline {
            self.deadline < other.deadline
        } else {
            self.rang_priority() > other.rang_priority()
        }
    }

    fn rang_priority(&self) -> u32 {
        match self.task_type {
            TaskType::Small  => 0,
            TaskType::Medium => 1,
            TaskType::Long   => 2,
        }
    }
}

struct TaskSpawner {
    lambda: f64,
}

impl TaskSpawner {
    fn with_lambda(lambda: f64) -> TaskSpawner {
        TaskSpawner { lambda }
    }

    fn task_from_type(&self, task_type: TaskType, rng: &mut SmallRng) -> Task {
        let (exec, period) = match task_type {
            TaskType::Small  => (6.0, 6.0 / self.lambda),
            TaskType::Medium => (12.0, 12.0 / self.lambda),
            TaskType::Long   => (20.0, 20.0 / self.lambda),
        };
        let exec = erlang_for_mean(exec, rng);
        let period = erlang_for_mean(period, rng);

        Task {exec, period}
    }
}


struct System {
    spawns: Vec<Spawn>, // end is top
    tasks: Vec<SystemTask>, // end is top
    tasks_processed: u32,
    task_spawner: TaskSpawner,
    system_type: SystemType,
}

impl System {
    fn with_task_spawner_and_type(task_spawner: TaskSpawner, system_type: SystemType) -> System {
        let mut system = System { spawns: Vec::new(), tasks: Vec::new(),
            tasks_processed: 0, task_spawner, system_type};
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
        let cmp_by_deadline = match self.system_type {
            SystemType::RateMonotonic => false,
            SystemType::EarliestDeadlineFirst => true,
        };
        while index > 0 && (new_task.more_important_than(
            &self.tasks[index - 1], cmp_by_deadline)) { index -= 1 };
        self.tasks.insert(index, new_task);
    }

    fn simulate(&mut self, simulation_time: f64, rng: &mut SmallRng) -> Stats {
        println!("========== Simulation start ({}) of type {:?} ==========", simulation_time, self.system_type);
        const DEADLINE_MULTIPLIER: f64 = 4.0;
        let mut current_time = 0.0;
        // let mut iterations_count = 0;
        let mut deadlines_missed = 0;
        let mut processor_free_time = 0.0;
        let mut waiting_times = Vec::new();
        while current_time < simulation_time {
            // iterations_count += 1;
            let earliest_finish =
                if let Some(&SystemTask {left_to_run, ..}) = self.tasks.last() {
                    current_time + left_to_run
                } else { std::f64::INFINITY };
            let earliest_spawn = self.spawns.last().unwrap().time;

            if earliest_spawn < earliest_finish {
                let time_delta = earliest_spawn - current_time;
                current_time = earliest_spawn;
                if self.tasks.is_empty() {
                    processor_free_time += time_delta;
                }

                // Do the spawn and prepare another one in place of it
                let Spawn {task_type, ..} = self.spawns.pop().unwrap();
                let Task {exec, period} = self.task_spawner.task_from_type(task_type, rng);
                self.add_spawn(Spawn {
                    time: current_time + period,
                    task_type
                });
                // Advance progress on current task
                if let Some(task) = self.tasks.last_mut() {
                    task.left_to_run -= time_delta;
                }
                // Insert the new task
                self.add_system_task(SystemTask {
                    left_to_run: exec,
                    execution_time: exec,
                    deadline: current_time + DEADLINE_MULTIPLIER * exec,
                    task_type,
                    birth_time: current_time,
                });
            } else { // finish first
                current_time = earliest_finish;
                // There are sure to be at least 1 tasks because earliest_finish
                // wouldn't have been first otherwise.
                // We are sure that the current task has finished because we've
                // jumped onto its finish time.
                let SystemTask{deadline, birth_time, execution_time, ..} = self.tasks.pop().unwrap();
                let waiting_time = (current_time - birth_time) - execution_time;
                waiting_times.push(waiting_time);
                self.tasks_processed += 1;
                if current_time > deadline { // expired
                    deadlines_missed += 1;
                } else { // all good
                }
            }
        }
        println!("========== Simulation end ==========");
        println!("%%%%%%%%% STATS %%%%%%%%%");
        let system_load = 1.0 - processor_free_time / current_time;
        println!("{} tasks processed ({} successful)", self.tasks_processed,
            self.tasks_processed - deadlines_missed);
        println!("{} tasks left in the system", self.tasks.len());
        // println!("----- Accounting for leftovers...");
        deadlines_missed += self.tasks.iter()
            .filter(|task| task.deadline < current_time).count() as u32;
        println!("{} deadlines missed in total", deadlines_missed);
        self.tasks.iter().for_each(|task|
            waiting_times.push((current_time - task.birth_time) - task.execution_time));
        let average_waiting_time = 1.0 / waiting_times.len() as f64 * waiting_times.into_iter().sum::<f64>();

        Stats {
            deadlines_missed,
            system_load,
            average_waiting_time,
        }
    }
}

pub fn lab5_work() {
    let seed = [10,4,3,8, 7,9,8,10, 14,18,12,12, 14,15,16,17];
    let mut rng = SmallRng::from_seed(seed);
    const SIMULATION_TIME: f64 = 10000.0;

    const MIN_LAMBDA: f64 = 0.0;
    const MAX_LAMBDA: f64 = 0.9;
    const LAMBDA_STEP: f64 = 0.05;
    let lambdas: Vec<_> = std::iter::repeat(LAMBDA_STEP)
        .scan(MIN_LAMBDA, |acc, curr| {*acc += curr; Some(*acc)})
        .take_while(|&x| x <= MAX_LAMBDA)
        .collect();

    let mut deadlines_missed_rm = Vec::new();
    let mut system_load_rm = Vec::new();
    let mut average_waiting_times_rm = Vec::new();
    for lambda in lambdas.iter() {
        let Stats {deadlines_missed, system_load, average_waiting_time} = System::with_task_spawner_and_type(
                TaskSpawner::with_lambda(*lambda), SystemType::RateMonotonic)
            .simulate(SIMULATION_TIME, &mut rng);
        deadlines_missed_rm.push(deadlines_missed as f64);
        system_load_rm.push(system_load);
        average_waiting_times_rm.push(average_waiting_time);
    }
    println!();
    println!();
    println!();
    let mut deadlines_missed_edf = Vec::new();
    let mut system_load_edf = Vec::new();
    let mut average_waiting_times_edf = Vec::new();
    for lambda in lambdas.iter() {
        let Stats {deadlines_missed, system_load, average_waiting_time} = System::with_task_spawner_and_type(
                TaskSpawner::with_lambda(*lambda), SystemType::EarliestDeadlineFirst)
            .simulate(SIMULATION_TIME, &mut rng);
        deadlines_missed_edf.push(deadlines_missed as f64);
        system_load_edf.push(system_load as f64);
        average_waiting_times_edf.push(average_waiting_time);
    }

    let renderer = Renderer::with_width_height(1100, 600);
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&deadlines_missed_rm, Some(&lambdas));
        renderer.export_to("deadlines_missed_rm.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&deadlines_missed_edf, Some(&lambdas));
        renderer.export_to("deadlines_missed_edf.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&system_load_rm, Some(&lambdas));
        renderer.export_to("system_load_rm.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&system_load_edf, Some(&lambdas));
        renderer.export_to("system_load_edf.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&average_waiting_times_rm, Some(&lambdas));
        renderer.export_to("average_waiting_times_rm.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&average_waiting_times_edf, Some(&lambdas));
        renderer.export_to("average_waiting_times_edf.png");
    }

    //     let begin = Instant::now();
    //     let elapsed = begin.elapsed().as_micros();
    //     times.push(elapsed as f64);
}

struct Renderer {
    surface: ImageSurface,
    context: Context,
    width: i32,
    height: i32,
}

impl Renderer {
    fn with_width_height(width: i32, height: i32) -> Renderer {
        let surface = ImageSurface::create(Format::ARgb32, width, height).unwrap();
        let context = Context::new(&surface);
        context.set_font_size(20.0);
        Renderer {
            surface,
            context,
            width,
            height
        }
    }

    fn export_to(&self, filename: &str) {
        let mut file = File::create(filename).expect("Can't create file");
        self.surface.write_to_png(&mut file).expect("Can't write to png");
    }

    fn clear(&self) {
        self.context.set_source_rgb(1.0, 1.0, 1.0);
        self.context.paint();
    }

    fn draw_func(&self, ys: &Vec<f64>, xs: Option<&Vec<f64>>) {
        let (w, h) = (self.width as f64, self.height as f64);
        let (origin_x, origin_y) = (5.0, h / 2.0);
        let max_y = ys.iter().fold(std::f64::NEG_INFINITY, |a, &b| a.max(b));
        let scale_y = (h / 2.0) / max_y;
        let step_x = (w - origin_x) / ys.len() as f64;

        self.context.set_source_rgb(1.0, 0.0, 0.0);
        self.context.set_line_width(2.0);
        let mut x = origin_x;
        self.context.move_to(x, origin_y - ys[0] * scale_y);
        x += step_x;
        for y in ys.iter().skip(1) {
            let y = origin_y - y * scale_y; // y axis is inverted
            self.context.line_to(x, y);
            x += step_x;
        }
        self.context.stroke();

        // Draw Y axis hints
        self.context.set_source_rgb(0.0, 0.0, 1.0);
        let y_amount = 4 as f64;
        let step_y = h / 2.0 / (y_amount + 1.0);
        let y_step = max_y / (y_amount + 1.0);
        let mut y = y_step * y_amount;
        for i in 1..=4 {
            self.context.move_to(7.0, i as f64 * step_y);
            let mut text = y.to_string();
            text.truncate(5);
            self.context.show_text(&text);
            y -= y_step;
        }
        self.context.stroke();

        // Maybe draw X axis hints (expects them to be uniformly distributed)
        if let Some(xs) = xs {
            const AMOUNT: usize = 5;
            let chunk_size = xs.len() / AMOUNT;
            self.context.set_source_rgb(0.0, 0.0, 1.0);
            for (index, x) in xs.into_iter().enumerate().step_by(chunk_size) {
                self.context.move_to(5.0 + index as f64 * step_x, origin_y - 5.0);
                let mut x_str = x.to_string();
                x_str.truncate(3);
                self.context.show_text(&x_str);
            }
            self.context.stroke();
        }
    }

    fn draw_axis(&self) {
        let (w, h) = (self.width as f64, self.height as f64);
        let (origin_x, origin_y) = (5.0, h / 2.0);
        self.context.set_source_rgb(0.0, 0.0, 0.0);
        self.context.set_line_width(4.0);

        // Y axis
        self.context.move_to(origin_x, 0.0);
        self.context.line_to(origin_x, h);
        // X axis
        self.context.move_to(0.0, origin_y);
        self.context.line_to(w, origin_y);

        self.context.stroke();
    }
}
