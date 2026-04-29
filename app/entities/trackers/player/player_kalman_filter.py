import numpy as np

class PlayerKalmanFilter:
    def __init__(self, dt: float):
        self.dt = dt

        self.x = np.zeros((4, 1))

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.P = np.eye(4) * 500  # incertidumbre inicial

        self.Q = np.eye(4) * 0.1  # ruido del modelo
        self.R = np.eye(2) * 5.0  # ruido de medicion

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z: np.ndarray):
        z = z.reshape(2, 1)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x
    
    def mahalanobis_distance(self, observed_x: float, observed_y: float):
        H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        z = np.array([
            [observed_x],
            [observed_y]
        ])

        x_pred, P_pred = self.predict()

        if x_pred is None:
            return np.inf

        innovation = z - H @ x_pred
        S = H @ P_pred @ H.T + self.R
        S_inv = np.linalg.pinv(S)

        d2 = innovation.T @ S_inv @ innovation

        return float(np.sqrt(d2.item()))
