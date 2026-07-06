// Branch and Bound implementation
use crate::model::{Model, StandardForm, VarId, ConSense};
use crate::simplex::{self, OptimizationStatus, Solution};

struct Node {
    pub model: Model,
    pub depth: usize,
}

fn is_integer_val(val: f64) -> bool {
    (val - val.round()).abs() < 1e-6
}

fn is_integer_solution(sol: &Solution, model: &Model) -> bool {
    for (i, var) in model.variables.iter().enumerate() {
        if matches!(var.var_type, crate::model::VarType::Integer | crate::model::VarType::Binary) {
            let val = sol.variables[i];
            if !is_integer_val(val) {
                return false;
            }
        }
    }
    true
}

fn find_branching_variable(sol: &Solution, model: &Model) -> Option<usize> {
    let mut best_idx = None;
    let mut max_fractionality = -1.0;

    for (i, var) in model.variables.iter().enumerate() {
        if matches!(var.var_type, crate::model::VarType::Integer | crate::model::VarType::Binary) {
            let val = sol.variables[i];
            let dist = (val - val.round()).abs();
            if dist > 1e-6 && dist > max_fractionality {
                max_fractionality = dist;
                best_idx = Some(i);
            }
        }
    }
    best_idx
}

pub fn solve_milp(base_model: &Model) -> Solution {
    // 1. Internal normalization: convert Minimization to Maximization by cloning the model
    // and negating objective coefficients.
    let mut model = base_model.clone();
    let is_minimize = model.objective_sense == crate::model::ObjectiveSense::Minimize;
    if is_minimize {
        model.objective_sense = crate::model::ObjectiveSense::Maximize;
        for var in &mut model.variables {
            var.obj_coeff = -var.obj_coeff;
        }
    }

    // 2. Add explicit constraints for variable bounds since StandardForm doesn't handle them
    let var_infos: Vec<(usize, crate::model::VarType, String, f64, f64)> = model.variables.iter().enumerate().map(|(i, var)| {
        (i, var.var_type, var.name.clone(), var.lower_bound, var.higher_bound)
    }).collect();

    for (i, var_type, name, lower_bound, higher_bound) in var_infos {
        let var_id = VarId(i);
        if var_type == crate::model::VarType::Binary {
            model.add_constraint(
                &format!("bound_binary_ub_{}", name),
                vec![(var_id, 1.0)],
                ConSense::LessEqual,
                1.0,
            );
        } else {
            if higher_bound.is_finite() {
                model.add_constraint(
                    &format!("bound_ub_{}", name),
                    vec![(var_id, 1.0)],
                    ConSense::LessEqual,
                    higher_bound,
                );
            }
        }
        if lower_bound > 1e-9 {
            model.add_constraint(
                &format!("bound_lb_{}", name),
                vec![(var_id, 1.0)],
                ConSense::GreaterEqual,
                lower_bound,
            );
        }
    }

    // Initialize branch and bound
    let mut best_incumbent_val = f64::NEG_INFINITY;
    let mut best_solution: Option<Solution> = None;

    let root_node = Node {
        model,
        depth: 0,
    };
    let mut stack: Vec<Node> = vec![root_node];
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 10000;

    while let Some(node) = stack.pop() {
        iterations += 1;
        if iterations > MAX_ITERATIONS {
            break;
        }

        // Relaxation
        let lp_model = StandardForm::from(&node.model);
        let current_solution = simplex::solve(&lp_model);

        // Pruning
        // By Infeasibility
        if current_solution.status == OptimizationStatus::Infeasible {
            continue;
        }

        // Handle Unboundedness
        if current_solution.status == OptimizationStatus::Unbounded {
            return Solution {
                status: OptimizationStatus::Unbounded,
                objective_value: if is_minimize { f64::NEG_INFINITY } else { f64::INFINITY },
                variables: current_solution.variables,
            };
        }

        // By Bound
        if current_solution.objective_value <= best_incumbent_val {
            continue;
        }

        // Check integrality
        if is_integer_solution(&current_solution, &node.model) {
            best_incumbent_val = current_solution.objective_value;
            best_solution = Some(current_solution);
            continue;
        }

        // Branching
        if let Some(branch_var_idx) = find_branching_variable(&current_solution, &node.model) {
            let branch_val = current_solution.variables[branch_var_idx];
            let var_name = &node.model.variables[branch_var_idx].name;

            // Left Node: var <= floor(branch_val)
            let mut left_model = node.model.clone();
            left_model.add_constraint(
                &format!("branch_{}_le_{}", var_name, node.depth),
                vec![(VarId(branch_var_idx), 1.0)],
                ConSense::LessEqual,
                branch_val.floor(),
            );
            stack.push(Node {
                model: left_model,
                depth: node.depth + 1,
            });

            // Right Node: var >= ceil(branch_val)
            let mut right_model = node.model.clone();
            right_model.add_constraint(
                &format!("branch_{}_ge_{}", var_name, node.depth),
                vec![(VarId(branch_var_idx), 1.0)],
                ConSense::GreaterEqual,
                branch_val.ceil(),
            );
            stack.push(Node {
                model: right_model,
                depth: node.depth + 1,
            });
        }
    }

    if let Some(mut sol) = best_solution {
        if is_minimize {
            sol.objective_value = -sol.objective_value;
        }
        sol
    } else {
        Solution {
            status: OptimizationStatus::Infeasible,
            objective_value: 0.0,
            variables: vec![0.0; base_model.variables.len()],
        }
    }
}
