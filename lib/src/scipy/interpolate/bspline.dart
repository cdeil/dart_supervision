import 'dart:math' as math;
import 'spline_interpolator.dart';

/// Modern BSpline implementation inspired by scipy's BSpline class.
///
/// This class provides a more modern interface compared to the legacy
/// splprep/splev functions, following scipy's recommended approach.
class BSpline {
  final List<double> _t; // knots
  final List<List<double>> _c; // coefficients for each dimension
  final int _k; // degree
  final int _dimensions;
  final List<CubicSpline> _splines;

  BSpline._(this._t, this._c, this._k, this._dimensions, this._splines);

  /// Creates a BSpline from knots, coefficients, and degree.
  factory BSpline.fromKnotsAndCoeffs(
    List<double> t,
    List<List<double>> c,
    int k,
  ) {
    final dimensions = c.length;

    // Create parameter values for each dimension
    final parameterValues = List.generate(
      t.length - k - 1,
      (i) => i.toDouble() / (t.length - k - 2),
    );

    // Create cubic splines for each dimension
    final splines = <CubicSpline>[];
    for (int dim = 0; dim < dimensions; dim++) {
      // For this simplified implementation, we'll use the coefficients as y values
      // and create a parameter array
      final coeffs = c[dim];
      final params = parameterValues.take(coeffs.length).toList();
      splines.add(CubicSpline.fromPoints(params, coeffs));
    }

    return BSpline._(t, c, k, dimensions, splines);
  }

  /// Evaluates the B-spline at given parameter values.
  ///
  /// Returns a list where each element contains the coordinates for one dimension.
  List<List<double>> call(List<double> u) {
    final result = List.generate(_dimensions, (_) => <double>[]);

    for (int dim = 0; dim < _dimensions; dim++) {
      final values = _splines[dim].eval(u);
      result[dim].addAll(values);
    }

    return result;
  }

  /// Gets the knot vector.
  List<double> get t => List.from(_t);

  /// Gets the coefficients for all dimensions.
  List<List<double>> get c => _c.map((dim) => List<double>.from(dim)).toList();

  /// Gets the degree of the spline.
  int get k => _k;

  /// Gets the number of dimensions.
  int get dimensions => _dimensions;
}

/// Modern interface for creating parametric B-spline curves.
///
/// This function provides a modern alternative to splprep, following
/// scipy's recommended BSpline-based approach.
///
/// Parameters:
/// - [x]: List of arrays representing the curve coordinates.
/// - [w]: Weights for each data point (currently ignored).
/// - [u]: Parameter values (if null, will be computed automatically).
/// - [k]: Degree of the spline (default 3 for cubic).
/// - [s]: Smoothing condition (currently ignored).
/// - [bcType]: Boundary conditions (currently ignored).
///
/// Returns a tuple of (BSpline, parameter_values).
(BSpline, List<double>) makeSplprep(
  List<List<double>> x, {
  List<double>? w,
  List<double>? u,
  int k = 3,
  double s = 0.0,
  String bcType = 'not-a-knot',
}) {
  if (x.isEmpty) {
    throw ArgumentError('Input arrays cannot be empty');
  }

  final dimensions = x.length;
  final numPoints = x[0].length;

  // Validate input
  for (int i = 1; i < dimensions; i++) {
    if (x[i].length != numPoints) {
      throw ArgumentError('All coordinate arrays must have the same length');
    }
  }

  if (numPoints < 2) {
    throw ArgumentError('Need at least 2 points to create a spline');
  }

  // Generate parameter values if not provided
  final parameterValues = u ?? _generateParameterValues(x);

  // Create coefficients (for this implementation, use input points)
  final coefficients = List.generate(
    dimensions,
    (i) => List<double>.from(x[i]),
  );

  // Generate knot vector
  final knots = _generateKnotVector(parameterValues, k);

  // Create BSpline
  final splines = <CubicSpline>[];
  for (int dim = 0; dim < dimensions; dim++) {
    splines.add(CubicSpline.fromPoints(parameterValues, x[dim]));
  }

  final bspline = BSpline._(knots, coefficients, k, dimensions, splines);

  return (bspline, parameterValues);
}

/// Generates a knot vector for the given parameter values and degree.
List<double> _generateKnotVector(List<double> u, int k) {
  final n = u.length;
  final knots = <double>[];

  // Add k+1 knots at the beginning
  for (int i = 0; i <= k; i++) {
    knots.add(u.first);
  }

  // Add interior knots
  for (int i = 1; i < n - 1; i++) {
    knots.add(u[i]);
  }

  // Add k+1 knots at the end
  for (int i = 0; i <= k; i++) {
    knots.add(u.last);
  }

  return knots;
}

/// Generates parameter values for the curve based on cumulative chord length.
///
/// This is the same function used in parametric_spline.dart, but we need it here too.
List<double> _generateParameterValues(List<List<double>> x) {
  final numPoints = x[0].length;
  final dimensions = x.length;

  if (numPoints == 1) return [0.0];

  final parameterValues = <double>[0.0];
  double totalLength = 0.0;

  // Calculate cumulative chord lengths
  for (int i = 1; i < numPoints; i++) {
    double segmentLength = 0.0;

    // Calculate Euclidean distance between consecutive points
    for (int dim = 0; dim < dimensions; dim++) {
      final diff = x[dim][i] - x[dim][i - 1];
      segmentLength += diff * diff;
    }
    segmentLength = math.sqrt(segmentLength);

    totalLength += segmentLength;
    parameterValues.add(totalLength);
  }

  // Normalize to [0, 1]
  if (totalLength > 0) {
    for (int i = 0; i < parameterValues.length; i++) {
      parameterValues[i] /= totalLength;
    }
  }

  return parameterValues;
}
