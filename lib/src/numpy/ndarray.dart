import 'dart:math' as math;
import 'dart:typed_data';

/// A multi-dimensional array class inspired by NumPy's ndarray.
///
/// This implementation uses a stride-based approach for memory layout
/// and supports view semantics, boolean indexing, and various data types.
class NDArray {
  final List<int> _shape;
  final List<int> _strides;
  final Float64List _data;
  final int _offset;
  final Type _dtype;

  /// Creates an NDArray with the given shape and data.
  NDArray._(this._shape, this._strides, this._data, this._offset, this._dtype);

  /// Factory constructor to create an NDArray from a shape and optional data.
  factory NDArray(List<int> shape, {List<double>? data, Type dtype = double}) {
    final size = shape.fold<int>(1, (a, b) => a * b);
    final actualData =
        data != null ? Float64List.fromList(data) : Float64List(size);

    if (data != null && data.length != size) {
      throw ArgumentError(
        'Data length ${data.length} does not match shape size $size',
      );
    }

    final strides = _computeStrides(shape);
    return NDArray._(List.from(shape), strides, actualData, 0, dtype);
  }

  /// Creates an NDArray filled with zeros.
  factory NDArray.zeros(List<int> shape, {Type dtype = double}) {
    return NDArray(shape, dtype: dtype);
  }

  /// Creates an NDArray filled with ones.
  factory NDArray.ones(List<int> shape, {Type dtype = double}) {
    final size = shape.fold<int>(1, (a, b) => a * b);
    return NDArray(shape, data: List.filled(size, 1.0), dtype: dtype);
  }

  /// Creates an NDArray from a nested list.
  factory NDArray.fromList(List<dynamic> list, {Type dtype = double}) {
    final shape = _inferShape(list);
    final flatData = _flattenList(list);
    return NDArray(shape, data: flatData, dtype: dtype);
  }

  /// Creates an identity matrix.
  factory NDArray.eye(int n, {Type dtype = double}) {
    final data = List<double>.filled(n * n, 0.0);
    for (int i = 0; i < n; i++) {
      data[i * n + i] = 1.0;
    }
    return NDArray([n, n], data: data, dtype: dtype);
  }

  /// Creates an NDArray with values from start to stop.
  factory NDArray.arange(double start, double stop, [double step = 1.0]) {
    if (step == 0) throw ArgumentError('Step cannot be zero');

    final length = ((stop - start) / step).ceil();
    if (length <= 0) return NDArray([0]);

    final data = List.generate(length, (i) => start + i * step);
    return NDArray([length], data: data);
  }

  /// Creates an NDArray with evenly spaced values.
  factory NDArray.linspace(double start, double stop, int num) {
    if (num <= 0) return NDArray([0]);
    if (num == 1) return NDArray([1], data: [start]);

    final step = (stop - start) / (num - 1);
    final data = List.generate(num, (i) => start + i * step);
    return NDArray([num], data: data);
  }

  /// Computes strides for row-major (C-style) layout.
  static List<int> _computeStrides(List<int> shape) {
    final strides = List<int>.filled(shape.length, 1);
    for (int i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /// Infers the shape of a nested list.
  static List<int> _inferShape(List<dynamic> list) {
    final shape = <int>[];
    dynamic current = list;

    while (current is List && current.isNotEmpty) {
      shape.add(current.length);
      current = current[0];
    }

    return shape;
  }

  /// Flattens a nested list into a 1D list.
  static List<double> _flattenList(List<dynamic> list) {
    final result = <double>[];

    void flatten(dynamic item) {
      if (item is List) {
        for (final element in item) {
          flatten(element);
        }
      } else {
        result.add((item as num).toDouble());
      }
    }

    flatten(list);
    return result;
  }

  /// Returns the shape of the array.
  List<int> get shape => List.unmodifiable(_shape);

  /// Returns the strides of the array.
  List<int> get strides => List.unmodifiable(_strides);

  /// Returns the data type of the array.
  Type get dtype => _dtype;

  /// Returns the number of dimensions.
  int get ndim => _shape.length;

  /// Returns the total number of elements.
  int get size => _shape.fold<int>(1, (a, b) => a * b);

  /// Returns the number of bytes per element.
  int get itemsize => 8; // Assuming double (64-bit)

  /// Returns the total number of bytes.
  int get nbytes => size * itemsize;

  /// Gets an element at the specified indices.
  double operator [](List<int> indices) {
    if (indices.length != _shape.length) {
      throw ArgumentError(
        'Number of indices ${indices.length} does not match dimensions ${_shape.length}',
      );
    }

    for (int i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= _shape[i]) {
        throw RangeError(
          'Index ${indices[i]} is out of bounds for dimension $i with size ${_shape[i]}',
        );
      }
    }

    final index = _offset + _computeFlatIndex(indices);
    return _data[index];
  }

  /// Sets an element at the specified indices.
  void operator []=(List<int> indices, double value) {
    if (indices.length != _shape.length) {
      throw ArgumentError(
        'Number of indices ${indices.length} does not match dimensions ${_shape.length}',
      );
    }

    for (int i = 0; i < indices.length; i++) {
      if (indices[i] < 0 || indices[i] >= _shape[i]) {
        throw RangeError(
          'Index ${indices[i]} is out of bounds for dimension $i with size ${_shape[i]}',
        );
      }
    }

    final index = _offset + _computeFlatIndex(indices);
    _data[index] = value;
  }

  /// Computes the flat index from multi-dimensional indices.
  int _computeFlatIndex(List<int> indices) {
    int flatIndex = 0;
    for (int i = 0; i < indices.length; i++) {
      flatIndex += indices[i] * _strides[i];
    }
    return flatIndex;
  }

  /// Returns a view of the array with a new shape.
  NDArray reshape(List<int> newShape) {
    final newSize = newShape.fold<int>(1, (a, b) => a * b);
    if (newSize != size) {
      throw ArgumentError(
        'Cannot reshape array of size $size into shape $newShape',
      );
    }

    final newStrides = _computeStrides(newShape);
    return NDArray._(List.from(newShape), newStrides, _data, _offset, _dtype);
  }

  /// Returns a transposed view of the array.
  NDArray transpose([List<int>? axes]) {
    if (ndim != 2 && axes == null) {
      throw ArgumentError('transpose() requires 2D array or explicit axes');
    }

    axes ??= [1, 0]; // Default transpose for 2D

    if (axes.length != ndim) {
      throw ArgumentError(
        'Number of axes ${axes.length} does not match dimensions $ndim',
      );
    }

    final newShape = axes.map((i) => _shape[i]).toList();
    final newStrides = axes.map((i) => _strides[i]).toList();

    return NDArray._(newShape, newStrides, _data, _offset, _dtype);
  }

  /// Returns a flattened 1D view of the array.
  NDArray flatten() {
    return reshape([size]);
  }

  /// Boolean indexing - returns elements where mask is true.
  NDArray operator &(NDArray mask) {
    if (!_shapeEquals(mask._shape, _shape)) {
      throw ArgumentError(
        'Boolean mask shape ${mask._shape} does not match array shape $_shape',
      );
    }

    final selectedIndices = <int>[];
    for (int i = 0; i < size; i++) {
      if (mask._data[mask._offset + i] != 0) {
        selectedIndices.add(i);
      }
    }

    final resultData = selectedIndices.map((i) => _data[_offset + i]).toList();
    return NDArray([selectedIndices.length], data: resultData, dtype: _dtype);
  }

  /// Element-wise addition.
  NDArray operator +(dynamic other) {
    if (other is num) {
      return _elementwiseScalar((a) => a + other.toDouble());
    } else if (other is NDArray) {
      return _elementwiseBinary(other, (a, b) => a + b);
    }
    throw ArgumentError('Unsupported operand type for +');
  }

  /// Element-wise subtraction.
  NDArray operator -(dynamic other) {
    if (other is num) {
      return _elementwiseScalar((a) => a - other.toDouble());
    } else if (other is NDArray) {
      return _elementwiseBinary(other, (a, b) => a - b);
    }
    throw ArgumentError('Unsupported operand type for -');
  }

  /// Element-wise multiplication.
  NDArray operator *(dynamic other) {
    if (other is num) {
      return _elementwiseScalar((a) => a * other.toDouble());
    } else if (other is NDArray) {
      return _elementwiseBinary(other, (a, b) => a * b);
    }
    throw ArgumentError('Unsupported operand type for *');
  }

  /// Element-wise division.
  NDArray operator /(dynamic other) {
    if (other is num) {
      return _elementwiseScalar((a) => a / other.toDouble());
    } else if (other is NDArray) {
      return _elementwiseBinary(other, (a, b) => a / b);
    }
    throw ArgumentError('Unsupported operand type for /');
  }

  /// Element-wise comparison operations.
  NDArray operator >(dynamic other) {
    if (other is num) {
      return _comparisonScalar((a) => a > other.toDouble());
    } else if (other is NDArray) {
      return _comparisonBinary(other, (a, b) => a > b);
    }
    throw ArgumentError('Unsupported operand type for >');
  }

  NDArray operator <(dynamic other) {
    if (other is num) {
      return _comparisonScalar((a) => a < other.toDouble());
    } else if (other is NDArray) {
      return _comparisonBinary(other, (a, b) => a < b);
    }
    throw ArgumentError('Unsupported operand type for <');
  }

  NDArray operator >=(dynamic other) {
    if (other is num) {
      return _comparisonScalar((a) => a >= other.toDouble());
    } else if (other is NDArray) {
      return _comparisonBinary(other, (a, b) => a >= b);
    }
    throw ArgumentError('Unsupported operand type for >=');
  }

  NDArray operator <=(dynamic other) {
    if (other is num) {
      return _comparisonScalar((a) => a <= other.toDouble());
    } else if (other is NDArray) {
      return _comparisonBinary(other, (a, b) => a <= b);
    }
    throw ArgumentError('Unsupported operand type for <=');
  }

  /// Applies a scalar operation element-wise.
  NDArray _elementwiseScalar(double Function(double) op) {
    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] = op(_data[_offset + i]);
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Applies a binary operation element-wise.
  NDArray _elementwiseBinary(
    NDArray other,
    double Function(double, double) op,
  ) {
    if (!_shapeEquals(_shape, other._shape)) {
      throw ArgumentError('Shape mismatch: $_shape vs ${other._shape}');
    }

    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] = op(_data[_offset + i], other._data[other._offset + i]);
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Returns a new array with each element rounded to the nearest integer.
  NDArray round() {
    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] = _data[_offset + i].roundToDouble();
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Returns a new array with each element clamped to the specified range.
  NDArray clamp(double min, double max) {
    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] = _data[_offset + i].clamp(min, max);
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Applies a comparison operation with scalar, returning boolean array.
  NDArray _comparisonScalar(bool Function(double) op) {
    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] = op(_data[_offset + i]) ? 1.0 : 0.0;
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Applies a comparison operation between arrays, returning boolean array.
  NDArray _comparisonBinary(NDArray other, bool Function(double, double) op) {
    if (!_shapeEquals(_shape, other._shape)) {
      throw ArgumentError('Shape mismatch: $_shape vs ${other._shape}');
    }

    final result = List<double>.filled(size, 0.0);
    for (int i = 0; i < size; i++) {
      result[i] =
          op(_data[_offset + i], other._data[other._offset + i]) ? 1.0 : 0.0;
    }
    return NDArray(_shape, data: result, dtype: _dtype);
  }

  /// Matrix multiplication.
  NDArray dot(NDArray other) {
    if (ndim != 2 || other.ndim != 2) {
      throw ArgumentError('dot() requires 2D arrays');
    }

    final m = _shape[0];
    final n = _shape[1];
    final p = other._shape[1];

    if (n != other._shape[0]) {
      throw ArgumentError('Matrix dimensions do not match for multiplication');
    }

    final result = List<double>.filled(m * p, 0.0);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < p; j++) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
          sum += this[[i, k]] * other[[k, j]];
        }
        result[i * p + j] = sum;
      }
    }

    return NDArray([m, p], data: result, dtype: _dtype);
  }

  /// Returns the sum of all elements.
  double sum() {
    double total = 0.0;
    for (int i = 0; i < size; i++) {
      total += _data[_offset + i];
    }
    return total;
  }

  /// Returns the mean of all elements.
  double mean() {
    return sum() / size;
  }

  /// Returns the maximum element.
  double max() {
    if (size == 0) throw StateError('Cannot find max of empty array');
    double maximum = _data[_offset];
    for (int i = 1; i < size; i++) {
      maximum = math.max(maximum, _data[_offset + i]);
    }
    return maximum;
  }

  /// Returns the minimum element.
  double min() {
    if (size == 0) throw StateError('Cannot find min of empty array');
    double minimum = _data[_offset];
    for (int i = 1; i < size; i++) {
      minimum = math.min(minimum, _data[_offset + i]);
    }
    return minimum;
  }

  /// Returns the square root of each element.
  NDArray sqrt() {
    return _elementwiseScalar((a) => math.sqrt(a));
  }

  /// Checks if two shapes are equal.
  static bool _shapeEquals(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Returns a copy of the array.
  NDArray copy() {
    final newData = Float64List.fromList(
      _data.sublist(_offset, _offset + size),
    );
    return NDArray(_shape, data: newData, dtype: _dtype);
  }

  /// Returns the array as a nested list.
  List<dynamic> toList() {
    if (ndim == 1) {
      return List.generate(size, (i) => _data[_offset + i]);
    }

    List<dynamic> buildNested(int depth, int offset) {
      if (depth == ndim - 1) {
        return List.generate(_shape[depth], (i) => _data[offset + i]);
      } else {
        final stride = _strides[depth];
        return List.generate(
          _shape[depth],
          (i) => buildNested(depth + 1, offset + i * stride),
        );
      }
    }

    return buildNested(0, _offset);
  }

  @override
  String toString() {
    if (size == 0) {
      return 'NDArray([], shape: $_shape)';
    }

    return 'NDArray(${toList()}, shape: $_shape)';
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! NDArray) return false;

    if (!_shapeEquals(_shape, other._shape)) return false;

    for (int i = 0; i < size; i++) {
      if ((_data[_offset + i] - other._data[other._offset + i]).abs() > 1e-10) {
        return false;
      }
    }

    return true;
  }

  @override
  int get hashCode {
    int hash = _shape.fold(0, (h, s) => h ^ s.hashCode);
    for (int i = 0; i < math.min(size, 10); i++) {
      hash ^= _data[_offset + i].hashCode;
    }
    return hash;
  }
}
