/// A cubic spline interpolator for smooth curve interpolation.
///
/// This implementation provides cubic spline interpolation similar to
/// what's used in scipy, with support for parametric curves.
class CubicSpline {
  final List<double> _x;
  final List<double> _y;
  final List<double> _coefficients;
  final int _n;

  CubicSpline._(this._x, this._y, this._coefficients, this._n);

  /// Creates a cubic spline from the given data points.
  ///
  /// Uses natural boundary conditions (second derivative = 0 at endpoints).
  factory CubicSpline.fromPoints(List<double> x, List<double> y) {
    if (x.length != y.length) {
      throw ArgumentError('x and y must have the same length');
    }
    if (x.length < 2) {
      throw ArgumentError('Need at least 2 points for interpolation');
    }

    final n = x.length;
    final coefficients = _computeCoefficients(x, y);

    return CubicSpline._(List.from(x), List.from(y), coefficients, n);
  }

  /// Evaluates the spline at the given points.
  List<double> eval(List<double> xEval) {
    return xEval.map((x) => _evaluateAt(x)).toList();
  }

  /// Evaluates the spline at a single point.
  double _evaluateAt(double x) {
    // Find the appropriate segment
    int segment = _findSegment(x);

    if (segment < 0) {
      // Extrapolate using first segment
      segment = 0;
    } else if (segment >= _n - 1) {
      // Extrapolate using last segment
      segment = _n - 2;
    }

    return _evaluateSegment(x, segment);
  }

  /// Finds the segment index for the given x value.
  int _findSegment(double x) {
    for (int i = 0; i < _n - 1; i++) {
      if (x >= _x[i] && x <= _x[i + 1]) {
        return i;
      }
    }
    return -1; // Not found
  }

  /// Evaluates the cubic polynomial in the given segment.
  double _evaluateSegment(double x, int segment) {
    final x0 = _x[segment];
    final x1 = _x[segment + 1];
    final y0 = _y[segment];
    final y1 = _y[segment + 1];
    final h = x1 - x0;

    if (h == 0) return y0;

    final t = (x - x0) / h;
    final m0 = _coefficients[segment];
    final m1 = _coefficients[segment + 1];

    // Cubic Hermite interpolation
    final h00 = 2 * t * t * t - 3 * t * t + 1;
    final h10 = t * t * t - 2 * t * t + t;
    final h01 = -2 * t * t * t + 3 * t * t;
    final h11 = t * t * t - t * t;

    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
  }

  /// Computes the cubic spline coefficients (derivatives at each point).
  static List<double> _computeCoefficients(List<double> x, List<double> y) {
    final n = x.length;
    final m = List.filled(n, 0.0);

    if (n == 2) {
      // Linear case
      final slope = (y[1] - y[0]) / (x[1] - x[0]);
      m[0] = slope;
      m[1] = slope;
      return m;
    }

    // Set up tridiagonal system for natural cubic spline
    final a = List.filled(n, 0.0);
    final b = List.filled(n, 0.0);
    final c = List.filled(n, 0.0);
    final d = List.filled(n, 0.0);

    // Interior points
    for (int i = 1; i < n - 1; i++) {
      final h0 = x[i] - x[i - 1];
      final h1 = x[i + 1] - x[i];
      final delta0 = (y[i] - y[i - 1]) / h0;
      final delta1 = (y[i + 1] - y[i]) / h1;

      a[i] = h0;
      b[i] = 2 * (h0 + h1);
      c[i] = h1;
      d[i] = 3 * (delta1 - delta0);
    }

    // Natural boundary conditions (second derivative = 0)
    b[0] = 1.0;
    c[0] = 0.0;
    d[0] = 0.0;

    a[n - 1] = 0.0;
    b[n - 1] = 1.0;
    d[n - 1] = 0.0;

    // Solve tridiagonal system using Thomas algorithm
    _solveTridiagonal(a, b, c, d, m);

    return m;
  }

  /// Solves a tridiagonal system using the Thomas algorithm.
  static void _solveTridiagonal(
    List<double> a,
    List<double> b,
    List<double> c,
    List<double> d,
    List<double> x,
  ) {
    final n = a.length;
    final cp = List.filled(n, 0.0);
    final dp = List.filled(n, 0.0);

    // Forward elimination
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
      final denominator = b[i] - a[i] * cp[i - 1];
      if (i < n - 1) {
        cp[i] = c[i] / denominator;
      }
      dp[i] = (d[i] - a[i] * dp[i - 1]) / denominator;
    }

    // Back substitution
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--) {
      x[i] = dp[i] - cp[i] * x[i + 1];
    }
  }
}

/// Generates linearly spaced values between start and stop.
///
/// Similar to numpy's linspace function.
List<double> linspace(double start, double stop, int num) {
  if (num <= 0) return [];
  if (num == 1) return [start];

  final result = <double>[];
  final step = (stop - start) / (num - 1);

  for (int i = 0; i < num; i++) {
    result.add(start + i * step);
  }

  return result;
}
