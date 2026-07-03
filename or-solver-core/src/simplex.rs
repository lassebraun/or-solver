// Simplex implementation
use crate::model::StandardForm;
use nalgebra::{DMatrix, DVector};

const EPSILON: f64 = 1e-8;

struct SimplexState {
    basis: Vec<usize>,
    non_basis: Vec<usize>,
    b_inv: DMatrix<f64>,
    x_b: DVector<f64>,
}

#[derive(Debug, PartialEq)]
pub enum OptimizationStatus {
    Optimal,
    Unbounded,
    Infeasible,
}

#[derive(Debug)]
pub struct Solution {
    pub status: OptimizationStatus,
    pub objective_value: f64,
    pub variables: Vec<f64>,
}

fn init_simplex(model: &StandardForm) -> SimplexState {
    // Assuming the origin is a valid solution
    let num_constraints = model.b_vector.nrows();
    let num_vars = model.a_matrix.ncols() - num_constraints;

    let basis: Vec<usize> = (num_vars..num_vars + num_constraints).collect();
    let non_basis: Vec<usize> = (0..num_vars).collect();
    let b_inv = DMatrix::identity(num_constraints, num_constraints);
    SimplexState {
        basis,
        non_basis,
        b_inv,
        x_b: model.b_vector.clone(),
    }
}

fn pricing(model: &StandardForm, state: &SimplexState) -> Option<usize> {
    let c_b = DVector::from_iterator(
        state.basis.len(),
        state.basis.iter().map(|&idx| model.c_vector[idx]),
    );
    let pi = c_b.transpose() * &state.b_inv;

    let mut best_reduced_cost = 0.0;
    let mut enter_idx: Option<usize> = None;

    for &j in &state.non_basis {
        let c_j = model.c_vector[j];

        let a_j = model.a_matrix.column(j);

        let pi_a_j = (&pi * a_j)[0];
        let reduced_cost = c_j - pi_a_j;

        if reduced_cost > best_reduced_cost + EPSILON {
            best_reduced_cost = reduced_cost;
            enter_idx = Some(j);
        }
    }
    enter_idx
}
fn ratio_test(state: &SimplexState, d: &DVector<f64>) -> Option<usize> {
    let mut min_ratio = f64::MAX;
    let mut leave_row: Option<usize> = None;

    for i in 0..state.basis.len() {
        // nalgebra erlaubt den 1D-Indexzugriff d[i], auch wenn es technisch eine Matrix ist
        // Alternativ: let d_i = d[(i, 0)];
        let d_i = d[i];

        if d_i > EPSILON {
            let ratio = state.x_b[i] / d_i;

            if ratio < min_ratio {
                min_ratio = ratio;
                leave_row = Some(i);
            }
        }
    }

    leave_row
}

fn basis_update(state: &mut SimplexState, leave_row: usize, enter_idx: usize, d: &DVector<f64>) {
    // switch indices
    let leave_idx = state.basis[leave_row];
    state.basis[leave_row] = enter_idx;

    if let Some(pos) = state.non_basis.iter().position(|&x| x == enter_idx) {
        state.non_basis[pos] = leave_idx;
    }

    let pivot_val = d[leave_row];
    let num_constraints = state.b_inv.nrows();

    let mut e_matrix: DMatrix<f64> = DMatrix::identity(num_constraints, num_constraints);

    for i in 0..num_constraints {
        if i == leave_row {
            e_matrix[(i, leave_row)] = 1.0 / pivot_val;
        } else {
            e_matrix[(i, leave_row)] = -d[i] / pivot_val;
        }
    }

    state.b_inv = &e_matrix * &state.b_inv;
    state.x_b = &e_matrix * &state.x_b;
}
fn extract_solution(
    model: &StandardForm,
    state: &SimplexState,
    status: OptimizationStatus,
) -> Solution {
    let num_constraints = model.b_vector.nrows();
    let num_vars = model.a_matrix.ncols() - num_constraints;

    let mut variables = vec![0.0; num_vars];

    for (row_idx, &var_idx) in state.basis.iter().enumerate() {
        if var_idx < num_vars {
            variables[var_idx] = state.x_b[row_idx];
        }
    }

    let mut objective_value = 0.0;
    for i in 0..num_vars {
        objective_value += model.c_vector[i] * variables[i];
    }

    Solution {
        status,
        objective_value,
        variables,
    }
}
pub fn run_simplex(model: &StandardForm) -> Solution {
    let mut state = init_simplex(model);

    while let Some(enter_idx) = pricing(model, &state) {
        let a_e = model.a_matrix.column(enter_idx);
        let d = &state.b_inv * a_e;

        if let Some(leave_row) = ratio_test(&state, &d) {
            basis_update(&mut state, leave_row, enter_idx, &d);
        } else {
            return extract_solution(&model, &state, OptimizationStatus::Unbounded);
        }
    }
    extract_solution(&model, &state, OptimizationStatus::Optimal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Model, ObjectiveSense, VarType, ConSense, StandardForm};

    #[test]
    fn test_simplex_less_equal() {
        let mut model = Model::new("Test", ObjectiveSense::Maximize);

        let x1 = model.add_var("x1", VarType::Integer, 0.0, f64::INFINITY, 5.0);
        let x2 = model.add_var("x2", VarType::Integer, 0.0, f64::INFINITY, 4.0);

        model.add_constraint(
            "Capacity A",
            vec![(x1, 1.0), (x2, 1.0)],
            ConSense::LessEqual,
            5.0
        );
        model.add_constraint(
            "Capacity B",
            vec![(x1, 10.0), (x2, 6.0)],
            ConSense::LessEqual,
            45.0
        );

        let mut standard_form_model = StandardForm::from(&model);
        let solution = run_simplex(&mut standard_form_model);

        assert_eq!(solution.status, OptimizationStatus::Optimal);
        assert!((solution.objective_value - 23.75).abs() < EPSILON);
        assert!((solution.variables[0] - 3.75).abs() < EPSILON);
        assert!((solution.variables[1] - 1.25).abs() < EPSILON);
    }
}

