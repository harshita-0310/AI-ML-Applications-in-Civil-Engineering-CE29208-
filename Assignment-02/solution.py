import numpy as np
import matplotlib.pyplot as plt


#defining functions for each problem
def solve_problem_1():
    """Concrete strength vs Cement content (Linear)"""
    print("\n--- Problem 1: Concrete Strength Prediction ---")
    C = np.array([300, 320, 340, 360, 400])
    fc = np.array([28.0, 30.0, 33.0, 35.0, 40.0])
    
    # Linear regression: fc = a0 + a1*C
    coeffs = np.polyfit(C, fc, 1)
    a1, a0 = coeffs
    
    prediction = a0 + a1 * 380
    print(f"Model: fc = {a0:.4f} + {a1:.4f}*C")
    print(f"Predicted fc for C=380: {prediction:.2f} MPa")
    plot_results(C, fc, coeffs, "Cement Content (kg/m3)", "Strength (MPa)", "Problem 1")

def solve_problem_2():
    """Young's Modulus (Linear Stress-Strain)"""
    print("\n--- Problem 2: Young's Modulus Estimation ---")
    strain = np.array([0.25, 0.50, 0.75, 1.00, 1.25]) * 1e-3
    stress = np.array([50, 100, 150, 205, 250])
    
    # Linear regression: stress = a + b*strain
    coeffs = np.polyfit(strain, stress, 1)
    b, a = coeffs
    
    print(f"Estimated Young's Modulus (E): {b/1000:.2f} GPa")
    plot_results(strain, stress, coeffs, "Strain", "Stress (MPa)", "Problem 2")

def solve_problem_3():
    """Beam Deflection (Linear) and Stiffness"""
    print("\n--- Problem 3: Beam Deflection and Stiffness ---")
    P = np.array([0, 10, 20, 30, 40])
    delta = np.array([0.05, 1.02, 2.01, 3.05, 4.02])
    
    coeffs = np.polyfit(P, delta, 1)
    a1, a0 = coeffs
    stiffness = 1/a1
    
    print(f"Model: delta = {a0:.4f} + {a1:.4f}*P")
    print(f"Estimated Stiffness (k = P/delta): {stiffness:.2f} kN/mm")
    plot_results(P, delta, coeffs, "Load (kN)", "Deflection (mm)", "Problem 3")

def solve_problem_4():
    """Concrete strength vs Cement content (Linear)"""
    print("\n--- Problem 4: Concrete Strength Prediction ---")
    C = np.array([300, 320, 340, 360, 400])
    fc = np.array([28.0, 30.0, 33.0, 35.0, 40.0])
    
    # Linear regression: fc = a0 + a1*C
    coeffs = np.polyfit(C, fc, 1)
    a1, a0 = coeffs
    
    prediction = a0 + a1 * 380
    print(f"Model: fc = {a0:.4f} + {a1:.4f}*C")
    print(f"Predicted fc for C=380: {prediction:.2f} MPa")
    plot_results(C, fc, coeffs, "Cement Content (kg/m3)", "Strength (MPa)", "Problem 4")



def solve_problem_5():
    """Multivariate Linear Regression for Concrete Strength"""
    print("\n--- Problem 5: Multivariate Linear Regression ---")
    C = np.array([300, 320, 340, 360, 380, 400])
    wc = np.array([0.55, 0.53, 0.50, 0.48, 0.46, 0.44])
    fc = np.array([20.5, 22.5, 26.0, 28.2, 31.5, 34.1])
    
    A = np.column_stack([np.ones(len(C)), C, wc])
    coeffs, _, _, _ = np.linalg.lstsq(A, fc, rcond=None)
    a0, a1, a2 = coeffs
    
    pred_fc = a0 + a1*350 + a2*0.49
    print(f"Model: fc = {a0:.4f} + {a1:.4f}*C + {a2:.4f}*(w/c)")
    print(f"Predicted fc for C=350, w/c=0.49: {pred_fc:.2f} MPa")

def solve_problem_6():
    """Quadratic Regression: Strength vs Age"""
    print("\n--- Problem 6: Quadratic Regression ---")
    t = np.array([7, 10, 14, 21, 28])
    fc = np.array([24, 28, 32, 38, 42])
    
    coeffs = np.polyfit(t, fc, 2)
    a2, a1, a0 = coeffs
    
    pred_18 = a2*(18**2) + a1*18 + a0
    print(f"Model: fc = {a0:.4f} + {a1:.4f}*t + {a2:.4f}*t^2")
    print(f"Predicted fc at 18 days: {pred_18:.2f} MPa")
    plot_results(t, fc, coeffs, "Age (days)", "Strength (MPa)", "Problem 6", poly_order=2)

def solve_logarithmic_problem(t, y, label_x, label_y, title, pred_t):
    """Helper for Problems 7, 8, 9, 11 (Logarithmic)"""
    ln_t = np.log(t)
    coeffs = np.polyfit(ln_t, y, 1)
    a1, a0 = coeffs
    prediction = a0 + a1 * np.log(pred_t)
    print(f"Model: {label_y} = {a0:.4f} + {a1:.4f}*ln(t)")
    print(f"Predicted value at t={pred_t}: {prediction:.2f}")

def solve_problem_7():
    print("\n--- Problem 7: Logarithmic Strength Growth ---")
    t = np.array([7, 10, 14, 21, 28])
    fc = np.array([24.0, 27.0, 30.0, 35.0, 38.0])
    solve_logarithmic_problem(t, fc, "Age (days)", "Strength (MPa)", "Problem 7", 18)

def solve_problem_8():
    print("\n--- Problem 8: Logarithmic Creep Strain ---")
    t = np.array([1, 3, 7, 14, 30])
    eps = np.array([120, 180, 240, 300, 360])
    solve_logarithmic_problem(t, eps, "Time (days)", "Creep Strain", "Problem 8", 10)

def solve_problem_9():
    print("\n--- Problem 9: Logarithmic Settlement ---")
    t = np.array([1, 2, 5, 10, 20])
    s = np.array([4.0, 5.2, 6.8, 8.0, 9.1])
    solve_logarithmic_problem(t, s, "Time (days)", "Settlement (mm)", "Problem 9", 15)

def solve_exponential_problem(t, y, label_x, label_y, title, pred_t):
    """Helper for Problems 10, 12 (Exponential)"""
    ln_y = np.log(y)
    coeffs = np.polyfit(t, ln_y, 1)
    B, lnA = coeffs
    A = np.exp(lnA)
    prediction = A * np.exp(B * pred_t)
    print(f"Model: {label_y} = {A:.4f} * e^({B:.4f}*t)")
    print(f"Predicted value at t={pred_t}: {prediction:.2f}")

def solve_problem_10():
    print("\n--- Problem 10: Exponential Strength Model ---")
    t = np.array([7, 10, 14, 21, 28])
    fc = np.array([20.0, 23.0, 27.0, 33.0, 38.0])
    solve_exponential_problem(t, fc, "Age (days)", "Strength (MPa)", "Problem 10", 18)

def solve_problem_11():
    print("\n--- Problem 11: Logarithmic Traffic Growth ---")
    t = np.array([1, 2, 4, 6, 10])
    adt = np.array([12000, 13700, 15500, 16500, 17800])
    solve_logarithmic_problem(t, adt, "Years", "ADT (veh/day)", "Problem 11", 8)

def solve_problem_12():
    print("\n--- Problem 12: Exponential Decay Amplitude ---")
    t = np.array([0, 1, 2, 3, 4])
    amp = np.array([10.0, 7.4, 5.5, 4.1, 3.0])
    solve_exponential_problem(t, amp, "Time (s)", "Amplitude (mm)", "Problem 12", 5)

def plot_results(x, y, coeffs, xlabel, ylabel, title, poly_order=1):
    plt.scatter(x, y, color='red', label='Data Points')
    p = np.poly1d(coeffs)
    x_line = np.linspace(min(x), max(x), 100)
    plt.plot(x_line, p(x_line), label='Regression Line')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


#asking user to choose a problem to be solved
def main():
    problems = {
        "1": solve_problem_1, "2": solve_problem_2, "3": solve_problem_3,
        "4": solve_problem_4, "5": solve_problem_5, "6": solve_problem_6,
        "7": solve_problem_7, "8": solve_problem_8, "9": solve_problem_9,
        "10": solve_problem_10, "11": solve_problem_11, "12": solve_problem_12
    }
    
    while True:
        print("\n--- Problem Set - 02 ---")
        print("Enter Problem Number (1-12)")
        choice = input("Choice: ").strip()
        
        if choice.lower() == 'q':
            break
        elif choice in problems:
            problems[choice]()
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
