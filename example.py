from MultivariableFunctionCalculator import FunctionCalculator

solver = FunctionCalculator(maximum=(25, 15), minimum=(-5, -12), saddle=(0, 0))
answer = solver.solve(n=1000)
print(answer)
