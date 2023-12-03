use plotters::prelude::*;

#[allow(dead_code)]
fn pdf(x: f64, y: f64) -> f64 {
    const SDX: f64 = 0.1;
    const SDY: f64 = 0.1;
    const A: f64 = 5.0;
    let x = x / 10.0;
    let y = y / 10.0;
    A * (-x * x / 2.0 / SDX / SDX - y * y / 2.0 / SDY / SDY).exp()
}

// Public function to demo plot
pub fn plot_demo_2d() -> Result<(), Box<dyn std::error::Error>> {

    // Configure chart backend
    let root = BitMapBackend::new("plot_2d.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root.margin(20, 20, 20, 20);

    // After this point, we should be able to construct a chart context
    let mut chart_context = ChartBuilder::on(&root)
        .caption("2d Plot", ("sans-serif", 40).into_font())
        .x_label_area_size(50) // Add buffer area to x axis
        .y_label_area_size(50) // Add buffer area to y axis
        .build_cartesian_2d(0f32..10f32, 0f32..100f32)?; // make a chart context

    // Build onto the chart context
    // configure_mesh is for 2d plots
    chart_context.configure_mesh().disable_mesh()
    .x_desc("x")
    .y_desc("y")
    .draw().unwrap();

    // Generate data
    let data = (0..=10)
    .map(|x| x as f32)
    .map(|x| (x, x * x)); // y = x^2

    let data2 = (0..=10)
    .map(|x| x as f32)
    .map(|x| (x, x*0.5)); // y = x/2

    // Add data to the chart
    chart_context.draw_series(LineSeries::new(data, &RED)).unwrap();
    chart_context.draw_series(LineSeries::new(data2, &BLUE)).unwrap();

    root.present()?;
    Ok(())
}

// Public function to demo plot
pub fn plot_opt_2d() -> Result<(), Box<dyn std::error::Error>> {

    // Configure chart backend
    let root = BitMapBackend::new("plot_2d_multimodal.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root.margin(20, 20, 20, 20);

    // After this point, we should be able to construct a chart context
    let mut chart_context = ChartBuilder::on(&root)
        .caption("2d Multimodel Plot", ("sans-serif", 40).into_font())
        .x_label_area_size(50) // Add buffer area to x axis
        .y_label_area_size(50) // Add buffer area to y axis
        .build_cartesian_2d(0f32..1.2f32, -1.5f32..2f32)?; // make a chart context

    // Build onto the chart context
    // configure_mesh is for 2d plots
    chart_context.configure_mesh().disable_mesh()
    .x_desc("x")
    .y_desc("y")
    .draw().unwrap();

    // Generate data
    let start: f32 = 0.0;
    let end: f32 = 1.2;
    let interval: f32 = 0.001;

    // Create a Vec<f32> with values spaced at 0.001 intervals
    let values = (0..=((end - start) / interval) as usize)
        .map(|i| (start + i as f32 * interval));

    let data = values.map(|x| (x, 1.6 * x * f32::sin(18.0*x))); // y = 1.6x*sin(18x)

    // Add data to the chart
    let pstyle = ShapeStyle {
        color: RGBAColor(0, 255, 0, 0.8),
        filled: false,
        stroke_width: 2,
    };
    chart_context.draw_series(LineSeries::new(data, pstyle)).unwrap();

    root.present()?;
    Ok(())
}
