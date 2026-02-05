"""
This script tests the UdpRawDataReader in various modes:
1. Real-time processing from UDP stream
2. Recording to .bin file while processing
3. Offline processing from recorded .bin file

Usage:
    # Real-time processing only
    python tools/capture_raw_data.py
    
    # Real-time with .bin recording (10 second duration)
    python tools/capture_raw_data.py --save-file test_capture.bin --duration 10
    
    # Offline processing from .bin file
    python tools/capture_raw_data.py --offline test_capture.bin
"""

import sys
import os
import time
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mwcore.radario.readers.TI.DCA1000EVM.udp_raw_reader import UdpRawDataReader
from mwcore.radario.readers.offlineReaders.raw_bin_reader import RawBinReader
from mwcore.signal_processing.radar_processor import RadarConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_realtime(save_file=None, duration=None):
    """
    Test real-time UDP capture and processing.
    
    Args:
        save_file: Optional path to save .bin file
        duration: Optional duration in seconds (runs indefinitely if None)
    """
    logger.info("=" * 60)
    logger.info("Testing UDP Raw Data Reader - Real-time Mode")
    logger.info("=" * 60)
    
    if save_file:
        logger.info(f"Will save raw data to: {save_file}")
    if duration:
        logger.info(f"Will run for {duration} seconds")
    else:
        logger.info("Press Ctrl+C to stop")
    
    # Create reader
    reader = UdpRawDataReader(
        save_to_file=save_file,
        enable_static_clutter_removal=True,
        energy_top_128=True,
        range_cut=True
    )
    
    try:
        # Start capture
        logger.info("Starting UDP capture...")
        reader.connect()
        logger.info("Capture started - waiting for radar data...")
        
        frame_count = 0
        start_time = time.time()
        last_log_time = start_time
        total_points = 0
        
        while True:
            # Check duration
            if duration and (time.time() - start_time) >= duration:
                logger.info(f"Duration {duration}s reached - stopping")
                break
            
            # Read frame
            data_ok, frame_num, det_obj = reader.read()
            
            if data_ok:
                frame_count += 1
                num_points = det_obj['numObj']
                total_points += num_points
                
                # Log every second
                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    avg_points = total_points / frame_count if frame_count > 0 else 0
                    
                    logger.info(
                        f"Frame {frame_num:4d} | {num_points:3d} points | "
                        f"FPS: {fps:.1f} | Avg points: {avg_points:.1f}"
                    )
                    last_log_time = current_time
            else:
                # No data available - small sleep to avoid busy waiting
                time.sleep(0.01)
        
        # Print summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Test completed:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Total time: {elapsed:.2f}s")
        logger.info(f"  Average FPS: {frame_count/elapsed:.2f}" if elapsed > 0 else "  Average FPS: N/A")
        logger.info(f"  Total points: {total_points}")
        logger.info(f"  Average points/frame: {total_points/frame_count:.1f}" if frame_count > 0 else "  Average points/frame: N/A")
        logger.info("=" * 60)
        
        if save_file:
            file_size_mb = os.path.getsize(save_file) / (1024 * 1024)
            logger.info(f"Saved {file_size_mb:.2f} MB to {save_file}")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        reader.close()


def test_offline(bin_file):
    """
    Test offline processing from a .bin file.
    
    Args:
        bin_file: Path to .bin file to process
    """
    logger.info("=" * 60)
    logger.info("Testing UDP Raw Data Reader - Offline Mode")
    logger.info("=" * 60)
    
    if not os.path.exists(bin_file):
        logger.error(f"File not found: {bin_file}")
        return
    
    file_size_mb = os.path.getsize(bin_file) / (1024 * 1024)
    logger.info(f"Reading from: {bin_file} ({file_size_mb:.2f} MB)")
    
    # Create offline reader
    reader = RawBinReader(
        file_path=bin_file,
        radar_config=RadarConfig()
    )
    
    try:
        frame_count = 0
        total_points = 0
        start_time = time.time()
        
        while True:
            # Read and process frame
            point_cloud = reader.read()
            
            if point_cloud is None:
                # End of file
                break
            
            frame_count += 1
            
            # point_cloud shape is (6, N) where N is number of points
            if point_cloud.size > 0:
                num_points = point_cloud.shape[1]
                total_points += num_points
                
                if frame_count % 10 == 0:
                    logger.info(f"Frame {frame_count:4d} | {num_points:3d} points")
            else:
                if frame_count % 10 == 0:
                    logger.info(f"Frame {frame_count:4d} | 0 points")
        
        # Print summary
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Offline processing completed:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Processing time: {elapsed:.2f}s")
        logger.info(f"  Processing rate: {frame_count/elapsed:.2f} frames/s" if elapsed > 0 else "  Processing rate: N/A")
        logger.info(f"  Total points: {total_points}")
        logger.info(f"  Average points/frame: {total_points/frame_count:.1f}" if frame_count > 0 else "  Average points/frame: N/A")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during offline processing: {e}", exc_info=True)
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test UDP Raw Data Reader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--offline',
        type=str,
        metavar='FILE',
        help='Process offline from .bin file instead of real-time UDP'
    )
    
    parser.add_argument(
        '--save-file',
        type=str,
        metavar='FILE',
        help='Save raw data to .bin file (real-time mode only)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        metavar='SECONDS',
        help='Duration to run in seconds (real-time mode only, runs indefinitely if not specified)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.offline and (args.save_file or args.duration):
        logger.error("--save-file and --duration are only valid in real-time mode")
        sys.exit(1)
    
    # Run appropriate test
    if args.offline:
        test_offline(args.offline)
    else:
        test_realtime(save_file=args.save_file, duration=args.duration)


if __name__ == '__main__':
    main()
