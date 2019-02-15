use rand::rngs::SmallRng;
// use std::iter;
use rand::SeedableRng;
use rand::Rng;
// use std::f64;

extern crate rand;

extern crate cairo;
use cairo::{ImageSurface, Format, Context};
use std::fs::File;

use std::time::{Duration, Instant};


fn dispersion(v: &Vec<f64>, mean: f64) -> f64 {
    v.iter().map(|x| (x - mean)*(x - mean)).sum::<f64>() / ((v.len() - 1) as f64)
}

fn mean(v: &Vec<f64>) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn func(amplitude: f64, frequency: f64, phase: f64, time: f64) -> f64 {
    amplitude * (frequency * time + phase).sin()
}

fn construct_signal(rng: &mut SmallRng,
                    harmonics_count: usize,
                    marginal_frequency: usize,
                    timespan: usize) -> Vec<f64> {
    const MAX_AMPLITUDE: f64 = 10.0;
    const MAX_PHASE:     f64 = 2.0 * std::f64::consts::PI;
    let frequency_step = marginal_frequency / harmonics_count;
    (frequency_step..).step_by(frequency_step).take(harmonics_count)
        .map(|frequency| {
            let frequency = frequency as f64;
            let amplitude = rng.gen::<f64>() * MAX_AMPLITUDE;
            let phase     = rng.gen::<f64>() * MAX_PHASE;
            (frequency, amplitude, phase)
        })
        .map(|(f, a, p)| (0..timespan).map(|t| func(a, f, p, t as f64)).collect::<Vec<_>>())
        .fold(vec![0.0; timespan], |sum, harmonic|
              sum.into_iter()
              .zip(harmonic.into_iter())
              .map(|(a, b)| a + b)
              .collect())
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

    fn draw_func(&self, ys: &Vec<f64>) {
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

fn main() {
    const HARMONICS_COUNT:    usize = 400;
    const MARGINAL_FREQUENCY: usize = 2700;
    const TIMESPAN:           usize = 64;
    let seed = [6,4,3,8, 7,9,8,10, 14,18,12,12, 14,15,16,17];
    let mut rng = SmallRng::from_seed(seed);

    let timespan_step = 32;
    let timespan_max = 1024;
    for timespan in (timespan_step..).step_by(timespan_step).take_while(|span| span <= &timespan_max) {
        let begin = Instant::now();
        let signal = construct_signal(&mut rng, HARMONICS_COUNT, MARGINAL_FREQUENCY, timespan);
        let elapsed = begin.elapsed().as_millis();
        println!("Elapsed for timespan={} : {}ms", timespan, elapsed);

        let mean = mean(&signal);
        let dispersion = dispersion(&signal, mean);
        println!("Mean = {}", mean);
        println!("Dispersion = {}", dispersion);

        if timespan == timespan_max / 4 {
            const WIDTH: i32 = 1100;
            const HEIGHT: i32 = 600;
            let renderer = Renderer::with_width_height(WIDTH, HEIGHT);
            renderer.clear();
            renderer.draw_axis();
            renderer.draw_func(&signal);
            renderer.export_to("out.png");
        }
    }
}
