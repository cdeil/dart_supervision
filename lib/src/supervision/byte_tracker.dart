import 'detections.dart';
import 'kalman_filter.dart';
import 'iou.dart';
import 'matching.dart';
import '../numpy/ndarray.dart';

/// ByteTracker implementation for multi-object tracking.
///
/// This is a simplified version of the ByteTracker algorithm,
/// similar to the supervision library's implementation.
class ByteTracker {
  final double trackActivationThreshold;
  final int lostTrackBuffer;
  final double minimumMatchingThreshold;
  final int frameRate;

  final KalmanFilter _kalmanFilter = KalmanFilter();
  final List<STrack> _trackedTracks = [];
  final List<STrack> _lostTracks = [];
  int _frameId = 0;

  ByteTracker({
    this.trackActivationThreshold = 0.25,
    this.lostTrackBuffer = 30,
    this.minimumMatchingThreshold = 0.8,
    this.frameRate = 30,
  });

  /// Updates tracker with new detections and returns tracked detections.
  Detections updateWithDetections(Detections detections) {
    _frameId++;

    if (detections.isEmpty) {
      // Mark all tracks as missed
      for (final track in _trackedTracks) {
        track.markMissed();
      }
      _removeDeletedTracks();
      return Detections.empty();
    }

    // Separate high and low confidence detections
    final highConfDetections = <int>[];
    final lowConfDetections = <int>[];

    for (int i = 0; i < detections.length; i++) {
      final conf = detections.confidence?[[i]] ?? 1.0;
      if (conf >= trackActivationThreshold) {
        highConfDetections.add(i);
      } else if (conf >= 0.1) {
        lowConfDetections.add(i);
      }
    }

    // Predict all tracks
    for (final track in [..._trackedTracks, ..._lostTracks]) {
      track.predict();
    }

    // First association: high confidence detections with confirmed tracks
    final confirmedTracks = _trackedTracks.where((t) => t.isConfirmed).toList();
    final (
      matches1,
      unmatchedTracks1,
      unmatchedDetections1,
    ) = _associateDetectionsToTracks(
      confirmedTracks,
      highConfDetections,
      detections,
    );

    // Update matched tracks
    for (final match in matches1) {
      final trackIdx = match[0];
      final detIdx = match[1];
      final track = confirmedTracks[trackIdx];
      final bbox = _getDetectionBbox(detections, highConfDetections[detIdx]);
      final conf = detections.confidence?[[highConfDetections[detIdx]]] ?? 1.0;
      track.update(bbox, conf);
    }

    // Second association: remaining high conf detections with lost tracks
    final activeLostTracks = _lostTracks.where((t) => !t.isDeleted).toList();
    final remainingHighConfDets = unmatchedDetections1
        .map((i) => highConfDetections[i])
        .toList();
    final (
      matches2,
      unmatchedTracks2,
      unmatchedDetections2,
    ) = _associateDetectionsToTracks(
      activeLostTracks,
      remainingHighConfDets,
      detections,
    );

    // Reactivate matched lost tracks
    for (final match in matches2) {
      final trackIdx = match[0];
      final detIdx = match[1];
      final track = activeLostTracks[trackIdx];
      final bbox = _getDetectionBbox(detections, remainingHighConfDets[detIdx]);
      final conf =
          detections.confidence?[[remainingHighConfDets[detIdx]]] ?? 1.0;
      track.update(bbox, conf);
      track.state = TrackState.confirmed;
      _trackedTracks.add(track);
    }

    // Remove reactivated tracks from lost tracks
    for (final match in matches2) {
      _lostTracks.remove(activeLostTracks[match[0]]);
    }

    // Third association: remaining detections with unconfirmed tracks
    final unconfirmedTracks = _trackedTracks
        .where((t) => t.isTentative)
        .toList();
    final finalUnmatchedDets = unmatchedDetections2
        .map((i) => remainingHighConfDets[i])
        .toList();
    final (
      matches3,
      unmatchedTracks3,
      unmatchedDetections3,
    ) = _associateDetectionsToTracks(
      unconfirmedTracks,
      finalUnmatchedDets,
      detections,
      threshold: 0.7,
    );

    // Update matched unconfirmed tracks
    for (final match in matches3) {
      final trackIdx = match[0];
      final detIdx = match[1];
      final track = unconfirmedTracks[trackIdx];
      final bbox = _getDetectionBbox(detections, finalUnmatchedDets[detIdx]);
      final conf = detections.confidence?[[finalUnmatchedDets[detIdx]]] ?? 1.0;
      track.update(bbox, conf);
    }

    // Mark unmatched tracks as missed
    for (final trackIdx in unmatchedTracks1) {
      final track = confirmedTracks[trackIdx];
      track.markMissed();
      if (track.isDeleted) {
        _trackedTracks.remove(track);
      } else {
        _lostTracks.add(track);
        _trackedTracks.remove(track);
      }
    }

    for (final trackIdx in unmatchedTracks3) {
      final track = unconfirmedTracks[trackIdx];
      track.markMissed();
      if (track.isDeleted) {
        _trackedTracks.remove(track);
      }
    }

    // Initialize new tracks from remaining detections
    final finalRemainingDets = unmatchedDetections3
        .map((i) => finalUnmatchedDets[i])
        .toList();
    for (final detIdx in finalRemainingDets) {
      final bbox = _getDetectionBbox(detections, detIdx);
      final conf = detections.confidence?[[detIdx]] ?? 1.0;
      if (conf >= trackActivationThreshold) {
        final track = STrack(bbox, conf, _kalmanFilter);
        track.frameId = _frameId;
        _trackedTracks.add(track);
      }
    }

    // Clean up deleted tracks
    _removeDeletedTracks();

    // Create output detections with tracker IDs
    return _createOutputDetections(detections);
  }

  /// Associates detections to tracks using IoU distance.
  (List<List<int>>, List<int>, List<int>) _associateDetectionsToTracks(
    List<STrack> tracks,
    List<int> detectionIndices,
    Detections detections, {
    double threshold = 0.5,
  }) {
    if (tracks.isEmpty || detectionIndices.isEmpty) {
      return (
        [],
        List.generate(tracks.length, (i) => i),
        List.generate(detectionIndices.length, (i) => i),
      );
    }

    // Build track bboxes
    final trackBboxes = NDArray([tracks.length, 4]);
    for (int i = 0; i < tracks.length; i++) {
      final xyxy = tracks[i].xyxy;
      for (int j = 0; j < 4; j++) {
        trackBboxes[[i, j]] = xyxy[[j]];
      }
    }

    // Build detection bboxes
    final detBboxes = NDArray([detectionIndices.length, 4]);
    for (int i = 0; i < detectionIndices.length; i++) {
      final detIdx = detectionIndices[i];
      for (int j = 0; j < 4; j++) {
        detBboxes[[i, j]] = detections.xyxy[[detIdx, j]];
      }
    }

    // Compute IoU distance matrix
    final costMatrix = IoUCalculator.iouDistance(trackBboxes, detBboxes);

    // Fuse with confidence scores if available
    NDArray fusedCosts = costMatrix;
    if (detections.confidence != null) {
      final scores = detectionIndices
          .map((i) => detections.confidence![[i]])
          .toList();
      fusedCosts = HungarianMatcher.fuseScore(costMatrix, scores);
    }

    // Solve assignment
    return HungarianMatcher.linearAssignment(fusedCosts, threshold);
  }

  /// Extracts bounding box for a detection.
  NDArray _getDetectionBbox(Detections detections, int index) {
    final bbox = NDArray([4]);
    final x1 = detections.xyxy[[index, 0]];
    final y1 = detections.xyxy[[index, 1]];
    final x2 = detections.xyxy[[index, 2]];
    final y2 = detections.xyxy[[index, 3]];

    // Convert to tlwh format
    bbox[[0]] = x1; // left
    bbox[[1]] = y1; // top
    bbox[[2]] = x2 - x1; // width
    bbox[[3]] = y2 - y1; // height

    return bbox;
  }

  /// Removes deleted tracks from internal lists.
  void _removeDeletedTracks() {
    _trackedTracks.removeWhere((track) => track.isDeleted);
    _lostTracks.removeWhere((track) => track.isDeleted);
  }

  /// Creates output detections with tracker IDs.
  Detections _createOutputDetections(Detections detections) {
    final activeTracks = _trackedTracks.where((t) => !t.isDeleted).toList();

    if (activeTracks.isEmpty) {
      return Detections.empty();
    }

    // Match detections to active tracks for output
    final outputIndices = <int>[];
    final outputTrackerIds = <double>[];

    for (int detIdx = 0; detIdx < detections.length; detIdx++) {
      final detBbox = NDArray(
        [4],
        data: [
          detections.xyxy[[detIdx, 0]],
          detections.xyxy[[detIdx, 1]],
          detections.xyxy[[detIdx, 2]],
          detections.xyxy[[detIdx, 3]],
        ],
      );

      double bestIou = 0.0;
      int bestTrackId = -1;

      for (final track in activeTracks) {
        final trackBbox = track.xyxy;
        final iou = IoUCalculator.iou(detBbox, trackBbox);
        if (iou > bestIou && iou > 0.3) {
          bestIou = iou;
          bestTrackId = track.trackId;
        }
      }

      if (bestTrackId != -1) {
        outputIndices.add(detIdx);
        outputTrackerIds.add(bestTrackId.toDouble());
      }
    }

    if (outputIndices.isEmpty) {
      return Detections.empty();
    }

    // Create filtered detections
    final indices = NDArray([
      outputIndices.length,
    ], data: outputIndices.map((i) => i.toDouble()).toList());
    final filtered = detections[indices];

    // Add tracker IDs
    final trackerIds = NDArray([
      outputTrackerIds.length,
    ], data: outputTrackerIds);

    return Detections(
      xyxy: filtered.xyxy,
      confidence: filtered.confidence,
      classId: filtered.classId,
      trackerId: trackerIds,
    );
  }

  /// Resets the tracker state.
  void reset() {
    _trackedTracks.clear();
    _lostTracks.clear();
    _frameId = 0;
    STrack.resetIdCounter();
  }

  @override
  String toString() {
    return 'ByteTracker(tracked: ${_trackedTracks.length}, lost: ${_lostTracks.length}, frame: $_frameId)';
  }
}
