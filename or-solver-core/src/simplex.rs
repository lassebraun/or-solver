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

    let basis: Vec<usize> = model.initial_basis.clone();

    let non_basis: Vec<usize> = (0..model.a_matrix.ncols()).filter(|i| !basis.contains(i)).collect();

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
    let num_vars = model.num_vars;

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
fn run_core_simplex(model: &StandardForm, state: &mut SimplexState) -> OptimizationStatus {
    while let Some(enter_idx) = pricing(model, &state) {
        let a_e = model.a_matrix.column(enter_idx);
        let d = &state.b_inv * a_e;

        if let Some(leave_row) = ratio_test(&state, &d) {
            basis_update(state, leave_row, enter_idx, &d);
        } else {
            return OptimizationStatus::Unbounded;
        }
    }
    OptimizationStatus::Optimal
}


pub fn solve(original_model: &StandardForm) -> Solution {
    let total_cols = original_model.a_matrix.ncols();
    let artificial_start_idx = original_model.num_vars + original_model.num_constraints;
    let has_artificial_vars = artificial_start_idx < total_cols;

    let mut state = init_simplex(original_model);


    // Phase 1
    if has_artificial_vars {
        let mut phase_1_model = original_model.clone();

        let mut phase_1_c = DVector::zeros(total_cols);
        for i in artificial_start_idx..total_cols {
            phase_1_c[i] = -1.0;
        }
        phase_1_model.c_vector = phase_1_c;

        run_core_simplex(&phase_1_model, &mut state);

        let mut phase_1_obj = 0.0;
        for (row_idx, &var_idx) in state.basis.iter().enumerate() {
            phase_1_obj += phase_1_model.c_vector[var_idx] * state.x_b[row_idx];
        }

        if phase_1_obj < -1e-6{
            return extract_solution(original_model, &state, OptimizationStatus::Infeasible);
        }
    }
    // Phase 2
    state.non_basis.retain(|&idx| idx < artificial_start_idx);

    let mut phase_2_model = original_model.clone();
    for i in artificial_start_idx..total_cols {
        phase_2_model.c_vector[i] = -1e-20;
    }

    let final_status = run_core_simplex(&phase_2_model, &mut state);

    extract_solution(original_model, &state, final_status)
}