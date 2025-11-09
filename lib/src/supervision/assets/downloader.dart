/// Asset downloader functionality
///
/// Dart implementation of supervision.assets.downloader module

import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;

import 'video_assets.dart';

/// Check if the MD5 hash of a file matches the original hash.
///
/// Parameters:
///   [filename]: The path to the file to be checked as a string.
///   [originalMd5Hash]: The original MD5 hash to compare against.
///
/// Returns:
///   `true` if the hashes match, `false` otherwise.
Future<bool> isMd5HashMatching(String filename, String originalMd5Hash) async {
  final file = File(filename);
  if (!await file.exists()) {
    return false;
  }

  try {
    final bytes = await file.readAsBytes();
    final digest = md5.convert(bytes);
    return digest.toString() == originalMd5Hash;
  } catch (e) {
    print('Error computing MD5 hash: $e');
    return false;
  }
}

/// Download a specified asset if it doesn't already exist or is corrupted.
///
/// Parameters:
///   [assetName]: The name or type of the asset to be downloaded.
///   [outputDir]: Optional directory to save the file. Defaults to current directory.
///   [onProgress]: Optional callback for download progress updates.
///
/// Returns:
///   The filename of the downloaded asset.
///
/// Example:
/// ```dart
/// import 'package:dart1/supervision/assets/assets.dart';
///
/// final filename = await downloadAssets(VideoAssets.vehicles);
/// print('Downloaded: $filename');
/// ```
Future<String> downloadAssets(
  dynamic assetName, {
  String? outputDir,
  void Function(int downloaded, int total)? onProgress,
}) async {
  String filename;

  if (assetName is VideoAssets) {
    filename = assetName.filename;
  } else if (assetName is String) {
    filename = assetName;
  } else {
    throw ArgumentError('assetName must be VideoAssets enum or String');
  }

  // Determine output path
  final outputPath =
      outputDir != null ? path.join(outputDir, filename) : filename;

  final file = File(outputPath);
  final metadata = videoAssetsMetadata[filename];

  if (metadata == null) {
    final validAssets = VideoAssets.allFilenames.join(', ');
    throw ArgumentError(
      'Invalid asset. It should be one of the following: $validAssets.',
    );
  }

  // Check if file exists and is valid
  if (await file.exists()) {
    if (await isMd5HashMatching(outputPath, metadata.md5Hash)) {
      print('$filename asset already exists and is valid.');
      return outputPath;
    } else {
      print('File corrupted. Re-downloading...');
      await file.delete();
    }
  }

  // Create output directory if needed
  await file.parent.create(recursive: true);

  print('Downloading $filename assets...');

  try {
    final response = await http.get(Uri.parse(metadata.url));

    if (response.statusCode != 200) {
      throw HttpException(
        'Failed to download $filename: HTTP ${response.statusCode}',
      );
    }

    final totalBytes = response.contentLength ?? response.bodyBytes.length;
    var downloadedBytes = 0;

    // Write file
    await file.writeAsBytes(response.bodyBytes);
    downloadedBytes = response.bodyBytes.length;

    // Report progress
    onProgress?.call(downloadedBytes, totalBytes);

    // Verify download
    if (await isMd5HashMatching(outputPath, metadata.md5Hash)) {
      print('$filename asset download complete.');
    } else {
      throw Exception('Downloaded file failed MD5 verification');
    }

    return outputPath;
  } catch (e) {
    print('Error downloading $filename: $e');
    if (await file.exists()) {
      await file.delete();
    }
    rethrow;
  }
}

/// Download a specified asset with streaming for large files and progress tracking.
///
/// This version uses streaming to handle large files more efficiently and provides
/// real-time progress updates.
Future<String> downloadAssetsStreaming(
  dynamic assetName, {
  String? outputDir,
  void Function(int downloaded, int total)? onProgress,
}) async {
  String filename;

  if (assetName is VideoAssets) {
    filename = assetName.filename;
  } else if (assetName is String) {
    filename = assetName;
  } else {
    throw ArgumentError('assetName must be VideoAssets enum or String');
  }

  // Determine output path
  final outputPath =
      outputDir != null ? path.join(outputDir, filename) : filename;

  final file = File(outputPath);
  final metadata = videoAssetsMetadata[filename];

  if (metadata == null) {
    final validAssets = VideoAssets.allFilenames.join(', ');
    throw ArgumentError(
      'Invalid asset. It should be one of the following: $validAssets.',
    );
  }

  // Check if file exists and is valid
  if (await file.exists()) {
    if (await isMd5HashMatching(outputPath, metadata.md5Hash)) {
      print('$filename asset already exists and is valid.');
      return outputPath;
    } else {
      print('File corrupted. Re-downloading...');
      await file.delete();
    }
  }

  // Create output directory if needed
  await file.parent.create(recursive: true);

  print('Downloading $filename assets...');

  try {
    final request = http.Request('GET', Uri.parse(metadata.url));
    final response = await request.send();

    if (response.statusCode != 200) {
      throw HttpException(
        'Failed to download $filename: HTTP ${response.statusCode}',
      );
    }

    final totalBytes = response.contentLength ?? 0;
    var downloadedBytes = 0;
    final sink = file.openWrite();

    try {
      await response.stream.listen(
        (chunk) {
          sink.add(chunk);
          downloadedBytes += chunk.length;
          onProgress?.call(downloadedBytes, totalBytes);
        },
        onDone: () {
          sink.close();
        },
        onError: (error) {
          sink.close();
          throw error;
        },
      ).asFuture();
    } finally {
      await sink.close();
    }

    // Verify download
    if (await isMd5HashMatching(outputPath, metadata.md5Hash)) {
      print('$filename asset download complete.');
    } else {
      throw Exception('Downloaded file failed MD5 verification');
    }

    return outputPath;
  } catch (e) {
    print('Error downloading $filename: $e');
    if (await file.exists()) {
      await file.delete();
    }
    rethrow;
  }
}
