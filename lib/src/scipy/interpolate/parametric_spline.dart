import 'dart:math' as math;
import 'spline_interpolator.dart';

/// Represents a parametric spline curve in N-dimensional space.
///
/// This is a simplified implementation inspired by scipy's splprep/splev
/// functionality, specifically designed for the smooth curve generation
/// used in object tracking visualization.
class ParametricSpline {
  final List<CubicSpline> _splines;
  final int _dimensions;

  ParametricSpline._(this._splines, this._dimensions);

  /// Evaluates the spline at the given parameter values.
  ///
  /// Returns a list where each element contains the coordinates for one dimension.
  /// For a 2D curve, returns [x_values, y_values].
  List<List<double>> eval(List<double> u) {
    final result = List.generate(_dimensions, (_) => <double>[]);

    for (int dim = 0; dim < _dimensions; dim++) {
      final values = _splines[dim].eval(u);
      result[dim].addAll(values);
    }

    return result;
  }
}

/// Creates a parametric spline representation of an N-dimensional curve.
///
/// This function is inspired by scipy's splprep and creates a smooth
/// parametric representation of a curve defined by a list of coordinate arrays.
///
/// Parameters:
/// - [x]: List of arrays representing the curve coordinates. For a 2D curve,
///        x[0] contains x-coordinates and x[1] contains y-coordinates.
/// - [s]: Smoothing condition (currently ignored in this implementation).
/// - [k]: Degree of the spline (currently ignored, uses cubic interpolation).
///
/// Returns a tuple of (ParametricSpline, List<double>) where the second element
/// contains the parameter values.
(ParametricSpline, List<double>) splprep(
  List<List<double>> x, {
  double s = 0.0,
  int k = 3,
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

  // Generate parameter values based on cumulative chord length
  final parameterValues = _generateParameterValues(x);

  // Create cubic splines for each dimension
  final splines = <CubicSpline>[];
  for (int dim = 0; dim < dimensions; dim++) {
    splines.add(CubicSpline.fromPoints(parameterValues, x[dim]));
  }

  final spline = ParametricSpline._(splines, dimensions);

  return (spline, parameterValues);
}

/// Evaluates a parametric spline at given parameter values.
///
/// This function mimics scipy's splev for parametric splines.
///
/// Parameters:
/// - [u]: Parameter values at which to evaluate the spline.
/// - [tck]: The spline representation (ParametricSpline).
///
/// Returns a list of coordinate arrays.
List<List<double>> splev(List<double> u, ParametricSpline tck) {
  return tck.eval(u);
}

/// Generates parameter values for the curve based on cumulative chord length.
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
