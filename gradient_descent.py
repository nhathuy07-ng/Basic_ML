import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import io
# gradient descent flow
## get f, initial value x0, learning_rate, threshold
## differentiate f with sympy
## update new_x = old_x - learning_rate * f'(old_x)
## if overshoot, reduce learning_rate

def log2text(l):
    return f"x: {round(l['cur_x'], 2)}, grad: {round(l['grad'], 2)}, iteration: {round(l['iter'], 2)}, learning rate: {round(l['rate'], 2)}"

def plot_func(sympy_func, logs, n_points):
    x_min = min([x['cur_x'] for x in logs])
    x_max = max([x['cur_x'] for x in logs])
    x_vals = np.linspace(x_min - 2, x_max + 2, n_points)
    y = [sympy_func.evalf(subs={"x": x}).__float__() for x in x_vals]
    


    # create a GIF

    frames = []

    for log in logs:
        fig, ax = plt.subplots(figsize=(6, 6))
        # init figure and axis
        ax.plot(x_vals, y, label=sympy_func, color="black")
        ax.legend()
        cx = log['cur_x']
        cy = sympy_func.evalf(subs={'x': cx})
        nx = log['next_x']
        ny = sympy_func.evalf(subs={'x': nx})
        ax.plot(cx, cy, 'ro')
        ax.plot(nx, ny, 'bo')
        ax.plot([cx,nx], [cy,ny], 'r-')
        
        ax.set_title(log2text(log))
        
        # write figure's buffer to imwrite
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close()
        
    
    imageio.mimsave("save.gif", frames, duration=1000/5)
    

def grad_descent(f: str, x0: float, learning_rate: float, iter: int):
    log = []

    equation = sympy.sympify(f)

    # differentiate w.r.t x
    diff = sympy.diff(equation, sympy.symbols('x'))
    print("DIFF: ", diff)

    eta = learning_rate;
    x = x0
    for i in range(iter):
        
        # diff at current x
        diff_x = diff.evalf(subs={"x": x})
        
        print("current x:", x, "grad:", diff_x, "iter:", iter, "eta: " , eta)
        
        x_new  = x - eta * diff_x

        log.append({"cur_x": x, "next_x": x_new, "grad": diff_x, "iter": i, "rate": eta})

        x = x_new


    return (equation, log)


f = input("f(x) = ")
x0 = float(input("x0 = "))
learning_rate = float(input("learning rate = "))
iter = float(input("iterations = "))

equation, log = grad_descent(f, x0, learning_rate, iter)
plot_func(equation, log, 1000)
