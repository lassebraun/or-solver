use or_solver_core::model::{Model, ObjectiveSense, VarType, ConSense, StandardForm};
use or_solver_core::simplex::solve;

fn main() {
    let mut model = Model::new("Test", ObjectiveSense::Maximize);

    let x1 = model.add_var("x1", VarType::Integer, 0.0, f64::INFINITY, 5.0);
    let x2 = model.add_var("x2", VarType::Integer, 0.0, f64::INFINITY, 4.0);

    // x1 + x2 <= 5
    model.add_constraint(
        "Capacity A",
        vec![(x1, 1.0), (x2, 1.0)],
        ConSense::LessEqual,
        5.0
    );
    // 10*x1 + 6*x2 <= 45
    model.add_constraint(
        "Capacity B",
        vec![(x1, 10.0), (x2, 6.0)],
        ConSense::LessEqual,
        45.0
    );

    model.add_constraint(
        "min",
        vec![(x1, 1.0), (x2, 1.0)],
        ConSense::GreaterEqual,
        1.0
    );
    println!("{:#?}", model);
    
    let mut standard_form_model = StandardForm::from(&model);
    let solution = solve(&mut standard_form_model, );
    
    println!("{:#?}", solution);
    
}
