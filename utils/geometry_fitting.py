import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.linear_model import LinearRegression


class LeafletGeometry():
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.single_point_names = ['R', 'L', 'N', 'RLC', 'RNC', 'LNC']

        self.RCI_points = self.landmarks['RCI']
        self.LCI_points = self.landmarks['LCI']
        self.NCI_points = self.landmarks['NCI']

        self.label_curves = [self.RCI_points, self.LCI_points, self.NCI_points]

    def calculate_bezier_curves(self, granularity=100):
        # Find control points
        self.RCI_left, self.MSE_RCI_left, self.RCI_right, self.MSE_RCI_right = split_and_approximate_curve(self.landmarks['RCI'], np.array(self.landmarks['R']))
        self.LCI_left, self.MSE_LCI_left, self.LCI_right, self.MSE_LCI_right = split_and_approximate_curve(self.landmarks['LCI'], np.array(self.landmarks['L']))
        self.NCI_left, self.MSE_NCI_left, self.NCI_right, self.MSE_NCI_right = split_and_approximate_curve(self.landmarks['NCI'], np.array(self.landmarks['N']))

        self.Control_points = [self.RCI_left, self.RCI_right, self.LCI_left, self.LCI_right,  self.NCI_left, self.NCI_right]
        self.MSE_list       = [self.MSE_RCI_left, self.MSE_RCI_right, self.MSE_LCI_left, self.MSE_LCI_right, self.MSE_NCI_left, self.MSE_NCI_right]
        # Sample Bezier Curve
        self.Bezier_RCI_left  = sample_bezier_curve_3d(self.RCI_left, granularity=granularity)
        self.Bezier_RCI_right = sample_bezier_curve_3d(self.RCI_right, granularity=granularity)
        self.Bezier_LCI_left  = sample_bezier_curve_3d(self.LCI_left, granularity=granularity)
        self.Bezier_LCI_right = sample_bezier_curve_3d(self.LCI_right, granularity=granularity)
        self.Bezier_NCI_left  = sample_bezier_curve_3d(self.NCI_left, granularity=granularity)
        self.Bezier_NCI_right = sample_bezier_curve_3d(self.NCI_right, granularity=granularity)

        self.Bezier_curves = [self.Bezier_RCI_left, self.Bezier_RCI_right, self.Bezier_LCI_left, self.Bezier_LCI_right, self.Bezier_NCI_left, self.Bezier_NCI_right]

    def plot(self, plot_control_points = True, plot_label_points = True, plot_bezier_curves = True, plot_single_points = True):

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        if plot_control_points:
            for control_points in self.Control_points:
                ax.scatter(control_points[1,0], control_points[1,1], control_points[1,2], color='orange', marker='o')

        if plot_label_points:
            for label_curve in self.label_curves:
                for point in label_curve:
                    ax.scatter(point[0], point[1], point[2], color='blue', marker='o')
        
        if plot_bezier_curves:
            for bezier_curve in self.Bezier_curves:
                ax.plot(bezier_curve[:,0], bezier_curve[:,1], bezier_curve[:,2], color='green')

        if plot_single_points:
            for key in self.landmarks:
                if key in self.single_point_names:
                    point = self.landmarks[key]
                    ax.scatter(point[0], point[1], point[2], color='red', marker='o')
                    ax.text(point[0], point[1], point[2], f'{key}', color='black')

        # Labels
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("3D Scatter Plot")

        # Show plot
        plt.show()

    def get_control_points(self):
        return self.Control_points
    
    def get_average_mse(self):
        return np.sum(np.array(self.MSE_list)) / len(self.MSE_list)


# Helper Functions
def split_and_approximate_curve(curve_points, middle_point):
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
    left_control_points, mse_left = fit_quadratic_bezier(left_curve)

    # Right Curve
    right_curve = curve[closest_index:]
    right_curve = np.vstack([middle_point, right_curve])
    right_control_points, mse_right = fit_quadratic_bezier(right_curve)

    return right_control_points, mse_left, left_control_points, mse_right

def bezier_quadratic(P0, P1, P2, t):
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

def sample_bezier_curve_3d(control_points, granularity=100):
    t_values = np.linspace(0, 1, granularity)
    return np.array([bezier_quadratic(control_points[0], control_points[1], control_points[2], t) for t in t_values])

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

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Azimuthal angle (radians)
    phi = np.arccos(z / r) if r != 0 else 0  # Elevation angle (radians)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates (r, θ, φ) to Cartesian coordinates (x, y, z)."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z

def draw_line_to_endpoints(points, granularity):
    return np.linspace(points[0], points[-1], granularity)