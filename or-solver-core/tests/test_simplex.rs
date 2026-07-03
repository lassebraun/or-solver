#[cfg(test)]
mod tests {
    use or_solver_core::model::*;
    use or_solver_core::simplex::*;

    #[test]
    fn test_simplex_paper_example() {

        // Feasible Model. Origin is a valid start
        let mut model = Model::new("Test", ObjectiveSense::Maximize);

        let x1 = model.add_var("x1", VarType::Integer, 0.0, f64::INFINITY, 5.0);
        let x2 = model.add_var("x2", VarType::Integer, 0.0, f64::INFINITY, 4.0);

        // x1 + x2 <= 5
        model.add_constraint(
            "Capacity A",
            vec![(x1, 1.0), (x2, 1.0)],
            ConSense::LessEqual,
            5.0,
        );
        // 10*x1 + 6*x2 <= 45
        model.add_constraint(
            "Capacity B",
            vec![(x1, 10.0), (x2, 6.0)],
            ConSense::LessEqual,
            45.0,
        );
        let mut standard_form_model_1 = StandardForm::from(&model);
        let solution = solve(&mut standard_form_model_1);

        assert_eq!(
            solution.status,
            OptimizationStatus::Optimal,
            "Solver should find the optimum"
        );

        let eps = 1e-6;

        assert!(
            (solution.objective_value - 23.75).abs() < eps,
            "Wrong objective Value! Expected: 23.75, got {}",
            solution.objective_value
        );

        assert!(
            (solution.variables[0] - 3.75).abs() < eps,
            "Wrong variables Value! Expected: 3.75, got {}",
            solution.variables[0]
        );

        assert!(
            (solution.variables[1] - 1.25).abs() < eps,
            "Wrong variables Value! Expected: 1.25, got {}",
            solution.variables[1]
        );

        // Add a constraint to make the origin infeasible
        // x_1 + x_2 >= 1
        model.add_constraint(
            "min",
            vec![(x1, 1.0), (x2, 1.0)],
            ConSense::GreaterEqual,
            1.0
        );
        let mut standard_form_model_2 = StandardForm::from(&model);
        let solution = solve(&mut standard_form_model_2);
        assert_eq!(
            solution.status,
            OptimizationStatus::Optimal,
            "Solver should find the optimum"
        );

        let eps = 1e-6;

        assert!(
            (solution.objective_value - 23.75).abs() < eps,
            "Wrong objective Value! Expected: 23.75, got {}",
            solution.objective_value
        );

        assert!(
            (solution.variables[0] - 3.75).abs() < eps,
            "Wrong variables Value! Expected: 3.75, got {}",
            solution.variables[0]
        );

        assert!(
            (solution.variables[1] - 1.25).abs() < eps,
            "Wrong variables Value! Expected: 1.25, got {}",
            solution.variables[1]
        );

    }

}
