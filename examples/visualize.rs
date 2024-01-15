use arroy::distances::Manhattan;
use arroy::{Database, Split, SplitFrame, Writer, VISUALIZE};
use heed::{EnvFlags, EnvOpenOptions};
use nannou::prelude::*;
use nannou_egui::{self, egui, Egui};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() {
    let mut rng = StdRng::seed_from_u64(62);
    // Open the environment with the appropriate flags.
    let flags = EnvFlags::empty();
    let mut env_builder = EnvOpenOptions::new();
    env_builder.map_size(1024 * 1024 * 1024 * 2);
    unsafe { env_builder.flags(flags) };
    let tempdir = tempfile::TempDir::new().unwrap();
    let env = env_builder.open(tempdir).unwrap();

    let mut wtxn = env.write_txn().unwrap();
    let database: Database<Manhattan> = env.create_database(&mut wtxn, None).unwrap();
    let writer = Writer::<Manhattan>::prepare(&mut wtxn, database, 0, 2).unwrap();

    for i in 0..1000 {
        writer
            .add_item(&mut wtxn, i, &[rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)])
            .unwrap();
    }
    writer.build(&mut wtxn, &mut rng, Some(1)).unwrap();
    drop(wtxn);

    nannou::app(model).update(update).run();
}

struct Settings {
    splits: Vec<Split>,
    displayed_split: usize,
    split_frame: usize,
    explode: usize,
    base_size: usize,
    weight_scaling: f32,
}

struct Model {
    settings: Settings,
    egui: Egui,
}

fn model(app: &App) -> Model {
    // Create window
    let window_id = app.new_window().view(view).raw_event(raw_window_event).build().unwrap();
    let window = app.window(window_id).unwrap();

    let egui = Egui::from_window(&window);

    Model {
        egui,
        settings: Settings {
            splits: unsafe { VISUALIZE.clone().unwrap() },
            displayed_split: 0,
            split_frame: 0,
            base_size: 5,
            weight_scaling: 0.1,
            explode: 550,
        },
    }
}

fn update(_app: &App, model: &mut Model, update: Update) {
    let egui = &mut model.egui;
    let settings = &mut model.settings;

    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();

    egui::Window::new("Settings").show(&ctx, |ui| {
        // Resolution slider
        ui.label("Splits:");
        ui.add(egui::Slider::new(&mut settings.displayed_split, 0..=settings.splits.len() - 1));
        ui.label("Split frame:");
        ui.add(egui::Slider::new(
            &mut settings.split_frame,
            0..=settings.splits[settings.displayed_split].splits.len() - 1,
        ));
        ui.label("Explode:");
        ui.add(egui::Slider::new(&mut settings.explode, 0..=1000));
        ui.label("Base size:");
        ui.add(egui::Slider::new(&mut settings.base_size, 0..=20));
        ui.label("Weight scaling:");
        ui.add(egui::Slider::new(&mut settings.weight_scaling, 0.01..=1.0));
    });
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    // Let egui handle things like keyboard and mouse input.
    model.egui.handle_raw_event(event);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let settings = &model.settings;
    let SplitFrame { left, right } =
        &settings.splits[settings.displayed_split].splits[settings.split_frame];

    let draw = app.draw();
    draw.background().color(BLACK);

    // the size of all circles in px
    draw.ellipse()
        .xy(Vec2::from(left.coords) * settings.explode as f32)
        .color(CYAN)
        .radius(settings.base_size as f32 * left.weight * settings.weight_scaling);
    draw.ellipse()
        .xy(Vec2::from(right.coords) * settings.explode as f32)
        .color(MAGENTA)
        .radius(settings.base_size as f32 * right.weight * settings.weight_scaling);
    draw.background().color(BLACK);

    for point in left.build_with.iter() {
        draw.ellipse()
            .xy(Vec2::from(*point) * settings.explode as f32)
            .color(DARKCYAN)
            .radius(settings.base_size as f32);
    }
    for point in right.build_with.iter() {
        draw.ellipse()
            .xy(Vec2::from(*point) * settings.explode as f32)
            .color(DARKMAGENTA)
            .radius(settings.base_size as f32);
    }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}
