from PEPit import PEP
from PEPit.functions import SmoothHypoconvexPLFunction

def wc_heavy_ball_hypoconvex_PL(mu, L, m, alpha, beta, n, wrapper="cvxpy", solver=None, verbose=1):
        # Instantiate PEP
    problem = PEP()

    # Declare a smooth strongly convex function
    func = problem.declare_function(SmoothHypoconvexPLFunction, L = L, m = m)

    # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
    xs = func.stationary_point()
    fs = func.value(xs)

    # Then define the starting point x0 of the algorithm as well as corresponding function value f0
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between f(x0) and f(x^*)
    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run one step of the heavy ball method
    x_new = x0
    x_old = x0

    for t in range(n):
        x_next = x_new - 1 / (L * (t + 2)) * func.gradient(x_new) + t / (t + 2) * (x_new - x_old)
        x_old = x_new
        x_new = x_next

    # Set the performance metric to the final distance to optimum
    problem.set_performance_metric(func.value(x_new) - fs)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(wrapper=wrapper, solver=solver, verbose=pepit_verbose)

    # Compute theoretical guarantee (for comparison)
    theoretical_tau = L / (2 * (n + 1))

    # Print conclusion if required
    if verbose != -1:
        print('*** Example file: worst-case performance of the Heavy-Ball method ***')
        print('\tPEPit guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(pepit_tau))
        print('\tTheoretical guarantee:\t f(x_n)-f_* <= {:.6} ||x_0 - x_*||^2'.format(theoretical_tau))

    # Return the worst-case guarantee of the evaluated method (and the reference theoretical value)
    return pepit_tau, theoretical_tau


if __name__ == "__main__":
    pepit_tau, theoretical_tau = wc_heavy_ball_hypoconvex_PL(L=1, n=5, wrapper="cvxpy", solver=None, verbose=1)