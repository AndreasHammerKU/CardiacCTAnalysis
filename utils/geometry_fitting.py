import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.linear_model import LinearRegression

def split_and_approximate_curve(ax, curve_points, middle_point, approximation):
    closest_index = 0
    for i, point in enumerate(curve_points):
        diff = np.array(point) - middle_point
        if i == 0:
            closest = diff
        elif np.linalg.norm(closest) > np.linalg.norm(diff):
            closest = diff
            closest_index = i
    curve = np.array(curve_points)
    
    # Left Curve
    left_curve = curve[:closest_index]
    left_curve = np.vstack([left_curve, middle_point])
    ax.scatter(left_curve[:,0], left_curve[:,1], left_curve[:,2], color='blue', marker='o')
    line = draw_line_to_endpoints(left_curve, 100)
    ax.plot(line[:,0], line[:,1], line[:,2], label=f'line of left curve')

    # Approximate bezier curve
    if approximation == 'bezier':
        curve_type = 'quadratic'
        control_points, mse = fit_quadratic_bezier(left_curve)
        print("left curve has mse: {}".format(mse))
        bezier_curve = sample_bezier_curve_3d(control_points, curve_type=curve_type)
        ax.plot(bezier_curve[:,0], bezier_curve[:,1], bezier_curve[:,2], label=f'Left Bezier curve for right leaflet')

        ax.scatter(control_points[1,0], control_points[1,1], control_points[1,2], color='orange', marker='o')
    elif approximation == 'polynomial':
        A, B, C, D, E, F, mse = fit_parabola(left_curve[:,0], left_curve[:,1], left_curve[:,2])
        print("loss of curve is {}".format(mse))
        plot_parabola_fit(ax, left_curve[:,0], left_curve[:,1], left_curve[:,2], A, B, C, D, E, F)
    elif approximation == 'parabola':
        A, B, C, D, E, F, G, H, mse = fit_parabola_curve(left_curve[:,0], left_curve[:,1], left_curve[:,2])
        print("loss of curve is {} for points \n{}".format(mse, left_curve))
        plot_parabola_curve(ax, left_curve[:,0], left_curve[:,1], left_curve[:,2], A, B, C, D, E, F, G, H)
    
    # Right Curve
    right_curve = curve[closest_index-1:]
    right_curve = np.vstack([middle_point, right_curve])
    ax.scatter(right_curve[:,0], right_curve[:,1], right_curve[:,2], color='green', marker='o')
    line = draw_line_to_endpoints(right_curve, 100)
    ax.plot(line[:,0], line[:,1], line[:,2], label=f'line of left curve')

    # Approximate Bezier Curve
    if approximation == 'bezier':
        control_points, mse = fit_quadratic_bezier(right_curve)
        print("left curve has mse: {}".format(mse))
        bezier_curve = sample_bezier_curve_3d(control_points, curve_type=curve_type)
        ax.plot(bezier_curve[:,0], bezier_curve[:,1], bezier_curve[:,2], label=f'Right Bezier curve for right leaflet')
        
        ax.scatter(control_points[1,0], control_points[1,1], control_points[1,2], color='orange', marker='o')
    elif approximation == 'polynomial':
        A, B, C, D, E, F, mse = fit_parabola(right_curve[:,0], right_curve[:,1], right_curve[:,2])
        print("loss of curve is {}".format(mse))
        plot_parabola_fit(ax, right_curve[:,0], right_curve[:,1], right_curve[:,2], A, B, C, D, E, F)
    elif approximation == 'parabola':
        A, B, C, D, E, F, G, H, mse = fit_parabola_curve(right_curve[:,0], right_curve[:,1], right_curve[:,2])
        print("loss of curve is {} for points \n{}".format(mse, right_curve))
        plot_parabola_curve(ax, right_curve[:,0], right_curve[:,1], right_curve[:,2], A, B, C, D, E, F, G, H)
    

def plot_3d_points(landmarks, approximation='bezier'):
    single_points = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    split_and_approximate_curve(ax, landmarks['RCI'], np.array(landmarks['R']), approximation)
    split_and_approximate_curve(ax, landmarks['LCI'], np.array(landmarks['L']), approximation)
    split_and_approximate_curve(ax, landmarks['NCI'], np.array(landmarks['N']), approximation)

    for key in landmarks:
        if key in single_points:
            point = landmarks[key]
            ax.scatter(point[0], point[1], point[2], color='red', marker='o')
            ax.text(point[0], point[1], point[2], f'{key}', color='black')

    # Labels
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Scatter Plot")

    # Show plot
    plt.show()

def bezier_quadratic(P0, P1, P2, t):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

def bezier_cubic(P0, P1, P2, P3, t):
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3

def sample_bezier_curve_3d(control_points, curve_type='quadratic', granularity=100):
    t_values = np.linspace(0, 1, granularity)
    curve_points = np.array([
        bezier_quadratic(control_points[0], control_points[1], control_points[2], t)
        if curve_type == 'quadratic' else
        bezier_cubic(control_points[0], control_points[1], control_points[2], control_points[3], t)
        for t in t_values
    ])
    return curve_points

def closest_t_on_bezier(bezier_func, control_points, Q):
    def distance_to_curve(t):
        B_t = bezier_func(*control_points, t)
        return np.linalg.norm(B_t - Q)
    
    res = minimize_scalar(distance_to_curve, bounds=(0, 1), method='bounded')
    return res.x

def fit_quadratic_bezier(points, lambda_reg =0.0001):
    P0, P2 = points[0], points[-1]
    
    def loss(P1_flat):
        P1 = np.array(P1_flat)
        control_points = (P0, P1, P2)
        t_values = np.array([closest_t_on_bezier(bezier_quadratic, control_points, Q) for Q in points])
        bezier_points = np.array([bezier_quadratic(*control_points, t) for t in t_values])

        mse_loss = (bezier_points - points).flatten()

        midpoint = (P0 + P2) / 2
        reg_loss = lambda_reg * np.linalg.norm(P1 - midpoint) ** 2

        return np.append(mse_loss, reg_loss)
    
    P1_init = (P0 + P2) / 2
    res = least_squares(loss, P1_init)
    fitted_control_points = np.array([P0, res.x, P2])
    
    mse = res.cost / len(points)

    return fitted_control_points, mse

def fit_cubic_bezier(points):
    P0, P3 = points[0], points[-1]
    
    def loss(P_flat):
        P1, P2 = np.split(np.array(P_flat), 2)
        control_points = (P0, P1, P2, P3)
        t_values = np.array([closest_t_on_bezier(bezier_cubic, control_points, Q) for Q in points])
        bezier_points = np.array([bezier_cubic(*control_points, t) for t in t_values])
        return (bezier_points - points).flatten()
    
    P1_init, P2_init = (2 * P0 + P3) / 3, (P0 + 2 * P3) / 3
    res = least_squares(loss, np.hstack([P1_init, P2_init]))
    fitted_control_points = np.array([P0, res.x[:3], res.x[3:], P3])
    
    mse = res.cost / len(points)  # Using least_squares cost directly
    
    return fitted_control_points, mse

def fit_quadratic_polynomial(points):
    t_values = np.linspace(0, 1, len(points))
    coeffs_x = np.polyfit(t_values, points[:, 0], 2)
    coeffs_y = np.polyfit(t_values, points[:, 1], 2)
    coeffs_z = np.polyfit(t_values, points[:, 2], 2)
    
    fitted_points = np.array([
        [np.polyval(coeffs_x, t), np.polyval(coeffs_y, t), np.polyval(coeffs_z, t)] for t in t_values
    ])
    mse = np.mean(np.linalg.norm(fitted_points - points, axis=1) ** 2)
    
    return fitted_points, mse

def fit_parabola(x, y, z):
    # Construct the design matrix
    X = np.column_stack((x**2, x*y, y**2, x, y, np.ones_like(x)))
    
    # Solve using least squares
    model = LinearRegression()
    model.fit(X, z)
    
    # Extract coefficients
    A, B, C, D, E, F = model.coef_[0], model.coef_[1], model.coef_[2], model.coef_[3], model.coef_[4], model.intercept_
    
    # Compute MSE
    z_pred = model.predict(X)
    mse = np.mean((z - z_pred)**2)
    
    return A, B, C, D, E, F, mse

def fit_parabola_curve(x, y, z):
    # Design matrix for quadratic regression
    X = np.column_stack((x**3, x**2, x, np.ones_like(x)))
    
    # Fit y = A x^2 + B x + C
    model_y = LinearRegression().fit(X, y)
    A, B, C, D = model_y.coef_[0], model_y.coef_[1], model_y.coef_[2], model_y.intercept_
    
    # Fit z = D x^2 + E x + F
    model_z = LinearRegression().fit(X, z)
    E, F, G, H = model_z.coef_[0], model_z.coef_[1], model_z.coef_[2], model_z.intercept_
    
    # Compute MSE
    y_pred = model_y.predict(X)
    z_pred = model_z.predict(X)
    mse_y = np.mean((y - y_pred) ** 2)
    mse_z = np.mean((z - z_pred) ** 2)
    mse_total = (mse_y + mse_z) / 2
    
    return A, B, C, D, E, F, G, H, mse_total

def plot_parabola_fit(ax, x, y, z, A, B, C, D, E, F):
    # Create a grid for plotting the fitted surface
    x_range = np.linspace(min(x), max(x), 30)
    y_range = np.linspace(min(y), max(y), 30)
    X, Y = np.meshgrid(x_range, y_range)
    Z = A*X**2 + B*X*Y + C*Y**2 + D*X + E*Y + F
    
    # Plot the surface
    ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan', edgecolor='k')

def plot_parabola_curve(ax, x, y, z, A, B, C, D, E, F, G, H):    
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = A * x_fit**3 + B * x_fit**2 + C * x_fit + D
    z_fit = E * x_fit**3 + F * x_fit**2 + G * x_fit + H
    
    # Plot the fitted curve
    ax.plot(x_fit, y_fit, z_fit, color='blue', linewidth=2, label='Fitted Parabola')

def draw_line_to_endpoints(points, granularity):
    return np.linspace(points[0], points[-1], granularity)