from simplex_method import simplex_method as sm

# How to input data exapmle:

# Objective Function: 12X_1 + 8X_2              --> [12, 8]
# S.T. (Solve To):  5X_1 + 2X_2 <= 150          --> [[5, 2, '<=', 150]]

objective_func = [3, 2, 5]
st = [[1, 2, 1, '<=', 43], 
      [3, 0, 2, '<=', 46],
      [1, 4, 0, '<=', 42]]

sm(objective_func, st, max=True)