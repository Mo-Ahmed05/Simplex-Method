import numpy as np

class simplex_method:

    m = 1e21    # Artificial Variable (Big M)

    def __init__(self, obj_func, st, max=True):
        self.obj_func = obj_func
        self.constraints = st
        self.max = max
        self.z_value = "Maximum" if self.max else "Minimum"
        
        # --- Run your setup functions ---
        self._build_standard_form()
        self._initialize_cj_and_cb()
        self._main_variables()
        self.solve()

    def _main_variables(self):
        vars_coeff = self.obj_func
        vars = {}

        for i, coeff in enumerate(vars_coeff):
            vars[f"X{i+1}"] = coeff

        self.main_vars = vars

    def _build_standard_form(self):

        constraints = self.constraints 
        tableau = []
        if not constraints:
            return []

        # Get the number of original decision variables
        num_orig_vars = len(constraints[0]) - 2

        total_new_vars = 0
        for constr in constraints:
            sign = constr[-2]
            if sign == '<=':
                total_new_vars += 1  # 1 slack variable
            elif sign == '>=':
                total_new_vars += 2  # 1 surplus, 1 artificial
            elif sign == '=':
                total_new_vars += 1  # 1 artificial variable
                
        total_cols = num_orig_vars + total_new_vars + 1

        # This pointer tracks which column the next new variable should go into
        current_new_var_col = num_orig_vars
        # This pointer tracks which column index has a artificial variable
        artificial_var_i = []
        # This pointer tracks which row index has a artificial variable
        artificial_var_row = []
        base_coeff_n = 0
        
        for i, constr in enumerate(constraints):
            # Create a new row, initialized to all zeros
            row = [0] * total_cols
            
            # Copy the original variable coefficients
            for j in range(num_orig_vars):
                row[j] = constr[j]
                
            # Set the RHS value
            row[-1] = constr[-1]
            
            # Get the sign to determine which new variables to add
            sign = constr[-2]
            
            if sign == '<=':
                row[current_new_var_col] = 1        # slack
                base_coeff_n += 1
                current_new_var_col += 1
                
            elif sign == '>=':
                row[current_new_var_col] = -1       # surplus
                current_new_var_col += 1

                row[current_new_var_col] = 1        # artificial
                base_coeff_n += 1
                artificial_var_i.append(current_new_var_col)
                artificial_var_row.append(i)
                current_new_var_col += 1
                
            elif sign == '=':
                row[current_new_var_col] = 1        # artificial
                base_coeff_n += 1
                artificial_var_i.append(current_new_var_col)
                artificial_var_row.append(i)
                current_new_var_col += 1
                
            tableau.append(row)
        
        self.tableau = np.array(tableau, dtype=float)
        self.artificial_var_i = artificial_var_i
        self.artificial_var_row = artificial_var_row
        self.base_coeff_n = base_coeff_n
        self.total_new_vars = total_new_vars # Save this for the next function

    def _initialize_cj_and_cb(self):
        m = self.m
        total_new_vars = self.total_new_vars
        artificial_var_i = self.artificial_var_i
        base_coeff_n = self.base_coeff_n
        artificial_var_row = self.artificial_var_row

        Cj = self.obj_func + [0]*total_new_vars

        for i in artificial_var_i:
            Cj[i] = -m if self.max else m    # setting the Big M in there indices

        Cj = np.array(Cj, dtype=float)

        Cb = np.zeros(base_coeff_n, dtype=float) # Use np.zeros for consistency

        for i in artificial_var_row:
            Cb[i] = -m if self.max else m

        self.Cj = Cj
        self.Cb = Cb

    def _pivot(self, pivot_row_i, pivot_col_i):
        pivot_element = self.tableau[pivot_row_i, pivot_col_i]
        self.tableau[pivot_row_i] = self.tableau[pivot_row_i] / pivot_element

        for i, row in enumerate(self.tableau):

            if i != pivot_row_i:
                row -= self.tableau[pivot_row_i] * row[pivot_col_i]

        self.Cb[pivot_row_i] = self.Cj[pivot_col_i]

    def solve(self):
        while True:
            
            Zj = np.array([0]*(len(self.Cj) + 1), dtype=float)

            for i, row in enumerate(self.tableau):
                Zj += self.Cb[i] * row    # Multiply each Cb by its row

            Cj_Zj = self.Cj - Zj[:-1] 
            
            # Check for optimality
            if (self.max and np.all(Cj_Zj <= 0)) or (not self.max and np.all(Cj_Zj >= 0)):
                
                # Check for infeasibility
                if (self.m in self.Cb) or (-self.m in self.Cb):
                    print("Solution is INFEASIBLE.")
                else:
                    # We need to define _print_solution
                    self._print_solution(Zj) 
                break
            
            # --- PIVOTING LOGIC ---
            
            # Find pivot column
            if self.max:
                pivot_col_i = np.argmax(Cj_Zj)
            else:
                pivot_col_i = np.argmin(Cj_Zj)

            # Find pivot row (Ratio Test)
            ratios = []
            for row in self.tableau:
                if row[pivot_col_i] > 0 and row[-1] >= 0:
                    ratio = row[-1] / row[pivot_col_i]
                    ratios.append(ratio)
                else:
                    ratios.append(np.inf)

            if np.all(np.isinf(ratios)):
                print('This problem has unbounded solution!')
                return

            pivot_row_i = np.argmin(ratios)
            
            # Pivot
            self._pivot(pivot_row_i, pivot_col_i)
    
    def _print_solution(self, Zj):
        print("Final Tableau:")
        print(self.tableau.round(2))
        print("\nFinal Cb:", self.Cb, self.main_vars)
        print(f"\nOptimal {self.z_value} Value:", Zj[-1].round(2))
                
        # Recalculate Cj-Zj just to be sure we have it
        Cj_Zj = self.Cj - Zj[:-1]

        # Find all basic variable columns
        basic_cols = []
        num_rows = self.tableau.shape[0]
        
        for j in range(len(self.Cj)): # Iterate through variable columns
            col = self.tableau[:, j]
            
            # A basic col has exactly one '1' and the rest are '0's
            is_basic_col = (np.sum(np.isclose(col, 1)) == 1) and (np.sum(np.isclose(col, 0)) == num_rows - 1)
            
            if is_basic_col:
                basic_cols.append(j)

        # Check Cj_Zj for zeros in NON-basic columns
        has_multi_optimal = False
        for j in range(len(Cj_Zj)):
            if j not in basic_cols:
                # Use np.isclose for floating point comparisons
                if np.isclose(Cj_Zj[j], 0):
                    has_multi_optimal = True
                    break
        
        if has_multi_optimal:
            print("\n*** This problem has Multi-Optimal (Alternative) Solutions. ***")