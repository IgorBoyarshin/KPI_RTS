use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::Rng;

extern crate rand;
extern crate cairo;
extern crate num;

use cairo::{ImageSurface, Format, Context};
use std::fs::File;
use std::time::{Instant};
use num::complex::Complex;


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

fn fast_fourier(signal: &Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    if signal.len() == 2 {
        return vec![signal[0] + signal[1], signal[0] - signal[1]]
    }

    let even = fast_fourier(&signal.iter().step_by(2).map(|x| *x).collect());
    let odd = fast_fourier(&signal.iter().skip(1).step_by(2).map(|x| *x).collect());
    #[allow(non_snake_case)]
    let N = signal.len();
    let w = |k| {
        if k % N == 0 { Complex::new(1.0, 0.0) }
        else {
            let arg = -2.0 * std::f64::consts::PI / N as f64 * k as f64;
            Complex::new(arg.cos(), arg.sin())
        }
    };
    let mut result = Vec::new();
    result.append(&mut even.iter().zip(odd.iter()).enumerate()
        .map(|(i, (e, o))| e + w(i) * o).collect::<Vec<_>>());
    result.append(&mut even.iter().zip(odd.iter()).enumerate()
        .map(|(i, (e, o))| e - w(i) * o).collect::<Vec<_>>());
    result
}

fn discrete_fourier(signal: &Vec<f64>) -> Vec<f64> {
    #[allow(non_snake_case)]
    let N = signal.len();
    // (0..N / 2).map(|p| {
    (0..N).map(|p| {
        let (re, im) = signal.iter().enumerate()
            .map(|(k, x)| {
                let arg = 2.0 * std::f64::consts::PI / N as f64 * p as f64 * k as f64;
                (x * arg.cos(), x * arg.sin())
            })
            .fold((0.0, 0.0), |(re, im), (next_re, next_im)| (re + next_re, im + next_im));
        (re*re + im*im).sqrt()
    })
    // .map(|f| f * 2.0 / N as f64)
    .collect()
}

fn func(amplitude: f64, frequency: f64, phase: f64, time: f64) -> f64 {
    amplitude * (frequency * time + phase).sin()
}

fn construct_signal(rng: &mut SmallRng,
                    harmonics_count: usize,
                    marginal_frequency: usize,
                    timespan: usize) -> Vec<f64> {
    const MAX_AMPLITUDE: f64 = 1.0;
    const MAX_PHASE:     f64 = 2.0 * std::f64::consts::PI;
    let frequency_step = marginal_frequency as f64 / harmonics_count as f64;
    std::iter::repeat(frequency_step).take(harmonics_count).scan(0.0, |acc, curr| {*acc += curr; Some(*acc)})
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

    fn draw_func(&self, ys: &Vec<f64>, xs: Option<Vec<usize>>) {
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
                self.context.show_text(&x.to_string());
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

pub fn lab4_work() {
    const HARMONICS_COUNT:    usize = 10; // n
    const MARGINAL_FREQUENCY: usize = 2700;
    const TIMESPAN:           usize = 8*256; // N
    let seed = [6,4,3,8, 7,9,8,10, 14,18,12,12, 14,15,16,17];
    let mut rng = SmallRng::from_seed(seed);

    const TIMESPAN_STEP: usize = 32;
    let mut times = Vec::new();
    // let timepoints: Vec<_> = (TIMESPAN_STEP..).step_by(TIMESPAN_STEP)
    //     .take_while(|&span| span <= TIMESPAN).collect();
    let timepoints: Vec<_> = std::iter::repeat(2)
        .scan(32, |acc, curr| {*acc *= curr; Some(*acc)})
        .take_while(|&x| x <= TIMESPAN)
        .collect();

    for timespan in timepoints.iter() {
        let signal_x = construct_signal(&mut rng, HARMONICS_COUNT, MARGINAL_FREQUENCY, *timespan);
        let begin = Instant::now();
            let _fast_fourier_x = fast_fourier(&signal_x.iter().map(|x| Complex::new(*x, 0.0)).collect());
        let elapsed = begin.elapsed().as_micros();
        times.push(elapsed as f64);
        // println!("Elapsed for timespan={} : {}micros", timespan, elapsed);
    }

    let signal_x = construct_signal(&mut rng, HARMONICS_COUNT, MARGINAL_FREQUENCY, TIMESPAN);
        let begin = Instant::now();
    let discrete_fourier_x = discrete_fourier(&signal_x);
        let elapsed = begin.elapsed().as_micros();
        println!("Elapsed for discrete: {}micros", elapsed);
        let begin = Instant::now();
    let fast_fourier_x = fast_fourier(&signal_x.iter().map(|x| Complex::new(*x, 0.0)).collect())
        .into_iter().map(|c| c.norm_sqr().sqrt()).collect();
        let elapsed = begin.elapsed().as_micros();
        println!("Elapsed for fast: {}micros", elapsed);

    let renderer = Renderer::with_width_height(1100, 600);
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&signal_x, None);
        renderer.export_to("signal_x.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&discrete_fourier_x, None);
        renderer.export_to("discrete_fourier.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&fast_fourier_x, None);
        renderer.export_to("fast_fourier.png");
    }
    {
        renderer.clear();
        renderer.draw_axis();
        renderer.draw_func(&times, Some(timepoints));
        renderer.export_to("times.png");
    }
}

