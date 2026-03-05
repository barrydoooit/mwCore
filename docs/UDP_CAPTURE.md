# UDP Packet Loss & Re-sync Logic

## Overview
The DCA1000 sends high-bandwidth raw ADC data over UDP, so packet loss is expected.

Old logic aborted capturing upon missing a packet, but because frames were reconstructed purely by accumulating packets, any dropped packet permanently misaligned the buffer by shifting the end of the current frame into the start of the next.

## New Re-sync Logic
The DCA1000 Hardware packet header provides an absolute `byte_count` of all bytes sent so far.
Because `bytes_in_frame` is constant, `(byte_count % bytes_in_frame)` gives the exact absolute position of the *current packet* within its frame structure.

When packets drop:
1. The script uses the absolute offset to pinpoint the exact array index where incoming data belongs.
2. It zero-fills the missing space and skips to the correct offset.
3. This guarantees that even if consecutive entire frames are lost, the buffer instantly recovers absolute synchronization the moment a new packet arrives.

## Current Bug: Corrupted Frames Escape to DSP
When a loss happens mid-frame, that frame is zero-filled with missing data. The DSP pipeline (`udp_raw_reader.py`) is designed to discard frames flagged as corrupted to prevent artifacting ("ghosts").

However, a bug in `udp_capture.py` prevents this flag from working. 

**The Bug:**
When `_frame_receiver` detects a loss, it correctly flags the buffer:
`self.lost_packet_flags[self.next_cap_buffer_position] = True`

But when the corrupted frame finishes assembling, `_store_frame` unconditionally clears the flag for the *entire* buffer slot just before saving it:
`self.lost_packet_flags[self.next_cap_buffer_position] = False`

**The Fix:**
Hold a local `frame_corrupted` boolean during the `while self.running:` loop, pass it to `_store_frame()`, and apply the flag based on the actual frame's integrity rather than hardcoding `False`.
