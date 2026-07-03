use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarType {
    Continuos,
    Integer,
    Binary,
}
#[derive(Debug, Clone)]
pub struct Variable {
    pub name: String,
    pub var_type: VarType,
    pub lower_bound: f64,
    pub higher_bound: f64,
    pub obj_coeff: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConSense {
    LessEqual,    // <=
    Equal,        // ==
    GreaterEqual, // >=
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: String,
    pub sense: ConSense,
    pub rhs: f64, //Right-Hand Side
    pub terms: Vec<(VarId, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveSense {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub objective_sense: ObjectiveSense,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, Clone)]
pub struct StandardForm {
    pub a_matrix: DMatrix<f64>,
    pub b_vector: DVector<f64>,
    pub c_vector: DVector<f64>,
    pub initial_basis: Vec<usize>,
    pub num_vars: usize,
    pub num_constraints: usize,
}

impl StandardForm {
    pub fn get_column(&self, idx: usize) -> DVector<f64> {
        self.a_matrix.column(idx).into_owned()
    }
    pub fn get_submatrix(&self, indices: &[usize]) -> DMatrix<f64> {
        self.a_matrix.select_columns(indices).into_owned()
    }
}

impl Model {
    pub fn new(name: &str, sense: ObjectiveSense) -> Self {
        Model{
            name: name.to_string(),
            objective_sense: sense,
            variables: Vec::new(),
            constraints: Vec::new()
        }
    }
    
    pub fn add_var(&mut self, name: &str, var_type: VarType, lower_bound: f64, higher_bound: f64, obj_coeff: f64) -> VarId {
        let id = VarId(self.variables.len());
        self.variables.push(Variable{
            name: name.to_string(),
            var_type,
            lower_bound,
            higher_bound,
            obj_coeff,
        });
        id
    }
    pub fn add_constraint(&mut self, name: &str, terms: Vec<(VarId, f64)>, sense: ConSense, rhs: f64) -> ConId {
        let id = ConId(self.constraints.len());
        self.constraints.push( Constraint {
            name: name.to_string(),
            sense,
            rhs,
            terms,
        });
        id
    }

}

impl From<&Model> for StandardForm {
    fn from(model: &Model) -> Self {
        let num_vars = model.variables.len();
        let num_constraints = model.constraints.len();

        // Count artificial variables needed for constraints other than <=
        let num_artificial = model.constraints.iter().filter( |c| {
            matches!(c.sense, ConSense::Equal | ConSense::GreaterEqual)
        }).count();


        let total_cols = num_vars + num_constraints + num_artificial;
        let mut a = DMatrix::zeros(num_constraints, total_cols);
        let mut b = DVector::zeros(num_constraints);
        let mut c = DVector::zeros(total_cols);

        let mut initial_basis = vec![0; num_constraints];

        for (i, var) in model.variables.iter().enumerate() {
            c[i] = var.obj_coeff;
        }

        let mut current_artificial_col = num_vars + num_constraints;

        for (row_idx, constraint) in model.constraints.iter().enumerate() {
            b[row_idx] = constraint.rhs;

            for &(var_id, coeff) in &constraint.terms {
                a[(row_idx, var_id.0)] = coeff;
            }
            let slack_col_idx = num_vars + row_idx;
            match constraint.sense {
                ConSense::LessEqual => {
                    a[(row_idx, slack_col_idx)] = 1.0;
                    initial_basis[row_idx] = slack_col_idx;
                },
                ConSense::Equal => {
                    a[(row_idx, slack_col_idx)] = 1.0;

                    initial_basis[row_idx] = current_artificial_col;
                    current_artificial_col += 1;
                },
                ConSense::GreaterEqual => {
                    a[(row_idx ,slack_col_idx)] = -1.0;
                    
                    a[(row_idx, current_artificial_col)] = 1.0;
                    
                    initial_basis[row_idx] = current_artificial_col;
                    current_artificial_col += 1;
                },
            }
        }
        StandardForm {
            a_matrix: a,
            b_vector: b,
            c_vector: c,
            initial_basis,
            num_vars,
            num_constraints,
        }
    }
}
