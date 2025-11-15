from simplex_method import simplex_method as sm

# How to input data exapmle:

# Objective Function: 12X_1 + 8X_2              --> [12, 8]
# S.T. (Solve To):  5X_1 + 2X_2 <= 150          --> [[5, 2, '<=', 150]]

objective_func = [12, 8]
st = [[5, 2, '<=', 150], 
      [2, 3, '<=', 100],
      [4, 2, '<=', 80]]

sm(objective_func, st, max=True)