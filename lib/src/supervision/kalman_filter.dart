import '../numpy/ndarray.dart';
import '../scipy/linalg/decomp_cholesky.dart';

/// Kalman filter for object tracking.
///
/// This is a simplified version focused on tracking bounding box centers
/// and velocities in 2D space, similar to the one used in ByteTracker.
class KalmanFilter {
  static const int _stateSize = 8; // [x, y, a, h, vx, vy, va, vh]
  static const int _measurementSize = 4; // [x, y, a, h]

  late NDArray _transitionMatrix;
  late NDArray _measurementMatrix;
  late NDArray _processNoise;
  late NDArray _measurementNoise;

  KalmanFilter() {
    _initializeMatrices();
  }

  void _initializeMatrices() {
    // State transition matrix (constant velocity model)
    _transitionMatrix = NDArray.eye(_stateSize);
    _transitionMatrix[[0, 4]] = 1.0; // x += vx
    _transitionMatrix[[1, 5]] = 1.0; // y += vy
    _transitionMatrix[[2, 6]] = 1.0; // a += va
    _transitionMatrix[[3, 7]] = 1.0; // h += vh

    // Measurement matrix (observe position and size)
    _measurementMatrix = NDArray.zeros([_measurementSize, _stateSize]);
    _measurementMatrix[[0, 0]] = 1.0; // observe x
    _measurementMatrix[[1, 1]] = 1.0; // observe y
    _measurementMatrix[[2, 2]] = 1.0; // observe a
    _measurementMatrix[[3, 3]] = 1.0; // observe h

    // Process noise covariance
    _processNoise = NDArray.eye(_stateSize);
    final positionStd = 1.0;
    final velocityStd = 1.0;

    _processNoise[[0, 0]] = positionStd * positionStd;
    _processNoise[[1, 1]] = positionStd * positionStd;
    _processNoise[[2, 2]] = positionStd * positionStd;
    _processNoise[[3, 3]] = positionStd * positionStd;
    _processNoise[[4, 4]] = velocityStd * velocityStd;
    _processNoise[[5, 5]] = velocityStd * velocityStd;
    _processNoise[[6, 6]] = velocityStd * velocityStd;
    _processNoise[[7, 7]] = velocityStd * velocityStd;

    // Measurement noise covariance
    _measurementNoise = NDArray.eye(_measurementSize);
    final measurementStd = 1.0;
    for (int i = 0; i < _measurementSize; i++) {
      _measurementNoise[[i, i]] = measurementStd * measurementStd;
    }
  }

  /// Initialize track with first measurement.
  (NDArray, NDArray) initiate(NDArray measurement) {
    if (measurement.size != _measurementSize) {
      throw ArgumentError('Measurement must have size $_measurementSize');
    }

    // Initial state: [x, y, a, h, 0, 0, 0, 0]
    final mean = NDArray([_stateSize]);
    mean[[0]] = measurement[[0]]; // x
    mean[[1]] = measurement[[1]]; // y
    mean[[2]] = measurement[[2]]; // a (aspect ratio)
    mean[[3]] = measurement[[3]]; // h (height)
    // velocities initialized to 0

    // Initial covariance (high uncertainty in velocities)
    final covariance = NDArray.eye(_stateSize);
    final positionVar = 200.0;
    final velocityVar = 10000.0;

    covariance[[0, 0]] = positionVar;
    covariance[[1, 1]] = positionVar;
    covariance[[2, 2]] = positionVar;
    covariance[[3, 3]] = positionVar;
    covariance[[4, 4]] = velocityVar;
    covariance[[5, 5]] = velocityVar;
    covariance[[6, 6]] = velocityVar;
    covariance[[7, 7]] = velocityVar;

    return (mean, covariance);
  }

  /// Predict next state.
  (NDArray, NDArray) predict(NDArray mean, NDArray covariance) {
    // x' = F * x
    final predictedMean = _transitionMatrix
        .dot(mean.reshape([_stateSize, 1]))
        .flatten();

    // P' = F * P * F^T + Q
    final temp = _transitionMatrix.dot(covariance);
    final predictedCovariance =
        temp.dot(_transitionMatrix.transpose()) + _processNoise;

    return (predictedMean, predictedCovariance);
  }

  /// Update state with measurement.
  (NDArray, NDArray) update(
    NDArray mean,
    NDArray covariance,
    NDArray measurement,
  ) {
    // Innovation: y = z - H * x
    final predictedMeasurement = _measurementMatrix
        .dot(mean.reshape([_stateSize, 1]))
        .flatten();
    final innovation = measurement - predictedMeasurement;

    // Innovation covariance: S = H * P * H^T + R
    final temp = _measurementMatrix.dot(covariance);
    final innovationCovariance =
        temp.dot(_measurementMatrix.transpose()) + _measurementNoise;

    // Kalman gain: K = P * H^T * S^-1
    // Using Cholesky decomposition for numerically stable solve
    final kalmanGain = covariance
        .dot(_measurementMatrix.transpose())
        .dot(
          _solvePositiveDefinite(
            innovationCovariance,
            NDArray.eye(innovationCovariance.shape[0]),
          ),
        );

    // Updated state: x = x + K * y
    final updatedMean =
        mean +
        kalmanGain.dot(innovation.reshape([_measurementSize, 1])).flatten();

    // Updated covariance: P = (I - K * H) * P
    final identity = NDArray.eye(_stateSize);
    final temp2 = kalmanGain.dot(_measurementMatrix);
    final updatedCovariance = (identity - temp2).dot(covariance);

    return (updatedMean, updatedCovariance);
  }

  /// Solve linear system Ax = b for positive definite matrix A using Cholesky decomposition.
  ///
  /// This is more numerically stable than matrix inversion for symmetric
  /// positive definite matrices commonly found in Kalman filters.
  NDArray _solvePositiveDefinite(NDArray matrix, NDArray rhs) {
    if (matrix.ndim != 2 || matrix.shape[0] != matrix.shape[1]) {
      throw ArgumentError('Matrix must be square');
    }

    try {
      // Try Cholesky decomposition first (most efficient for SPD matrices)
      final chol = LinAlg.choFactor(matrix, lower: true);
      return LinAlg.choSolve(chol, rhs);
    } on LinAlgError {
      // Fall back to Gauss-Jordan if Cholesky fails
      return _matrixInverseGaussJordan(matrix).dot(rhs);
    }
  }

  /// Fallback Gauss-Jordan elimination for matrix inversion.
  NDArray _matrixInverseGaussJordan(NDArray matrix) {
    if (matrix.ndim != 2 || matrix.shape[0] != matrix.shape[1]) {
      throw ArgumentError('Matrix must be square');
    }

    final n = matrix.shape[0];
    final augmented = NDArray([n, 2 * n]);

    // Create augmented matrix [A | I]
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        augmented[[i, j]] = matrix[[i, j]];
        augmented[[i, j + n]] = i == j ? 1.0 : 0.0;
      }
    }

    // Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
      // Find pivot
      double pivot = augmented[[i, i]];
      if (pivot.abs() < 1e-10) {
        throw ArgumentError('Matrix is singular');
      }

      // Scale row
      for (int j = 0; j < 2 * n; j++) {
        augmented[[i, j]] = augmented[[i, j]] / pivot;
      }

      // Eliminate column
      for (int k = 0; k < n; k++) {
        if (k != i) {
          final factor = augmented[[k, i]];
          for (int j = 0; j < 2 * n; j++) {
            augmented[[k, j]] = augmented[[k, j]] - factor * augmented[[i, j]];
          }
        }
      }
    }

    // Extract inverse matrix
    final inverse = NDArray([n, n]);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        inverse[[i, j]] = augmented[[i, j + n]];
      }
    }

    return inverse;
  }
}

/// Track state enumeration.
enum TrackState { tentative, confirmed, deleted }

/// Single object track.
///
/// Represents a single tracked object with Kalman filter state,
/// similar to ByteTracker's STrack class.
class STrack {
  static int _nextId = 1;

  /// Resets the global track ID counter.
  static void resetIdCounter() {
    _nextId = 1;
  }

  final int trackId;
  late NDArray _mean;
  late NDArray _covariance;
  TrackState state = TrackState.tentative;
  int frameId = -1;
  int timeSinceUpdate = 0;
  int hits = 1;
  double score = 0.0;

  final KalmanFilter _kalmanFilter;
  late NDArray _lastTlwh; // Store last known box

  STrack(NDArray tlwh, double confidence, this._kalmanFilter)
    : trackId = _nextId++,
      score = confidence {
    _lastTlwh = tlwh.copy();
    final xyah = _tlwhToXyah(tlwh);
    final result = _kalmanFilter.initiate(xyah);
    _mean = result.$1;
    _covariance = result.$2;
  }

  /// Convert top-left-width-height to center-x-center-y-aspect-height format.
  static NDArray _tlwhToXyah(NDArray tlwh) {
    final result = NDArray([4]);
    result[[0]] = tlwh[[0]] + tlwh[[2]] / 2; // center x
    result[[1]] = tlwh[[1]] + tlwh[[3]] / 2; // center y
    result[[2]] = tlwh[[2]] / tlwh[[3]]; // aspect ratio
    result[[3]] = tlwh[[3]]; // height
    return result;
  }

  /// Convert center-x-center-y-aspect-height to top-left-width-height format.
  static NDArray _xyahToTlwh(NDArray xyah) {
    final result = NDArray([4]);
    final w = xyah[[2]] * xyah[[3]];
    result[[0]] = xyah[[0]] - w / 2; // top-left x
    result[[1]] = xyah[[1]] - xyah[[3]] / 2; // top-left y
    result[[2]] = w; // width
    result[[3]] = xyah[[3]]; // height
    return result;
  }

  /// Predict track state to next frame.
  void predict() {
    final result = _kalmanFilter.predict(_mean, _covariance);
    _mean = result.$1;
    _covariance = result.$2;
    timeSinceUpdate++;
  }

  /// Update track with new detection.
  void update(NDArray tlwh, double confidence) {
    _lastTlwh = tlwh.copy();
    final xyah = _tlwhToXyah(tlwh);
    final result = _kalmanFilter.update(_mean, _covariance, xyah);
    _mean = result.$1;
    _covariance = result.$2;

    timeSinceUpdate = 0;
    hits++;
    score = confidence;

    if (state == TrackState.tentative && hits >= 3) {
      state = TrackState.confirmed;
    }
  }

  /// Mark track as deleted.
  void markMissed() {
    if (state == TrackState.tentative) {
      state = TrackState.deleted;
    } else if (timeSinceUpdate > 30) {
      state = TrackState.deleted;
    }
  }

  /// Get current bounding box in tlwh format.
  NDArray get tlwh {
    if (_mean.size < 4) {
      // If Kalman filter not initialized, return stored box
      return _lastTlwh.copy();
    }
    final xyah = NDArray(
      [4],
      data: [
        _mean[[0]],
        _mean[[1]],
        _mean[[2]],
        _mean[[3]],
      ],
    );
    return _xyahToTlwh(xyah);
  }

  /// Get current bounding box in xyxy format.
  NDArray get xyxy {
    final tlwhBox = tlwh;
    final result = NDArray([4]);
    result[[0]] = tlwhBox[[0]]; // x1
    result[[1]] = tlwhBox[[1]]; // y1
    result[[2]] = tlwhBox[[0]] + tlwhBox[[2]]; // x2
    result[[3]] = tlwhBox[[1]] + tlwhBox[[3]]; // y2
    return result;
  }

  bool get isConfirmed => state == TrackState.confirmed;
  bool get isDeleted => state == TrackState.deleted;
  bool get isTentative => state == TrackState.tentative;
}
