import math

def taylor_sin(x_rad, tol=1e-10, max_terms=50):
    term = x_rad
    sinx = 0.0
    n = 0
    
    while n < max_terms and abs(term) > tol:
        sinx += term
        n += 1
        term *= -1 * x_rad**2 / ((2*n)*(2*n+1)) 
        
    return sinx

# x = 30 degrees
x_deg = 30
x_rad = math.radians(x_deg)

approx = taylor_sin(x_rad)

print("sin(30°) using Taylor series:", approx)

