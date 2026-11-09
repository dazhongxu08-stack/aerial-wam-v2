package com.aerial.tellogs;

import android.media.MediaCodec;
import android.media.MediaFormat;
import android.util.Log;
import android.view.Surface;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Receive Tello H.264 UDP (port 11111) and decode to a Surface via MediaCodec. */
public final class TelloVideo implements AutoCloseable {
    private static final String TAG = "TelloVideo";

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicInteger frameSeq = new AtomicInteger(0);
    private final AtomicLong lastFrameAt = new AtomicLong(0L);
    private Thread thread;
    private DatagramSocket sock;
    private MediaCodec codec;
    private Surface surface;

    public int frameSeq() {
        return frameSeq.get();
    }

    public Double frameAgeS() {
        long t = lastFrameAt.get();
        if (t <= 0) return null;
        return (System.currentTimeMillis() - t) / 1000.0;
    }

    public synchronized void start(Surface surface) {
        stop();
        this.surface = surface;
        running.set(true);
        thread = new Thread(this::loop, "tello-video");
        thread.setDaemon(true);
        thread.start();
    }

    public synchronized void stop() {
        running.set(false);
        try {
            if (sock != null) sock.close();
        } catch (Exception ignored) {
        }
        sock = null;
        try {
            if (codec != null) {
                codec.stop();
                codec.release();
            }
        } catch (Exception ignored) {
        }
        codec = null;
        if (thread != null) {
            try {
                thread.join(800);
            } catch (InterruptedException ignored) {
            }
        }
        thread = null;
        frameSeq.set(0);
        lastFrameAt.set(0L);
    }

    @Override
    public void close() {
        stop();
    }

    private void loop() {
        try {
            sock = new DatagramSocket(null);
            sock.setReuseAddress(true);
            sock.setReceiveBufferSize(512 * 1024);
            sock.bind(new InetSocketAddress(TelloDirect.VIDEO_PORT));
            sock.setSoTimeout(1000);
            Log.i(TAG, "listening UDP :" + TelloDirect.VIDEO_PORT);

            codec = MediaCodec.createDecoderByType(MediaFormat.MIMETYPE_VIDEO_AVC);
            MediaFormat format = MediaFormat.createVideoFormat(
                    MediaFormat.MIMETYPE_VIDEO_AVC, 960, 720);
            codec.configure(format, surface, null, 0);
            codec.start();

            byte[] buf = new byte[65535];
            MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
            long lastLog = 0;
            int pkts = 0;

            while (running.get()) {
                // Drain output
                while (true) {
                    int out = codec.dequeueOutputBuffer(info, 0);
                    if (out >= 0) {
                        codec.releaseOutputBuffer(out, true);
                        if (info.size > 0) {
                            frameSeq.incrementAndGet();
                            lastFrameAt.set(System.currentTimeMillis());
                        }
                    } else {
                        break;
                    }
                }

                try {
                    DatagramPacket pkt = new DatagramPacket(buf, buf.length);
                    sock.receive(pkt);
                    pkts++;
                    int len = pkt.getLength();
                    if (len <= 0) continue;
                    int in = codec.dequeueInputBuffer(2000);
                    if (in < 0) continue;
                    ByteBuffer input = codec.getInputBuffer(in);
                    if (input == null) continue;
                    input.clear();
                    input.put(buf, 0, len);
                    codec.queueInputBuffer(in, 0, len, System.nanoTime() / 1000, 0);
                } catch (java.net.SocketTimeoutException e) {
                    long now = System.currentTimeMillis();
                    if (now - lastLog > 2000) {
                        Log.i(TAG, "waiting pkts=" + pkts + " seq=" + frameSeq.get());
                        lastLog = now;
                    }
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "video loop: " + e.getMessage(), e);
        } finally {
            try {
                if (codec != null) {
                    codec.stop();
                    codec.release();
                }
            } catch (Exception ignored) {
            }
            codec = null;
            try {
                if (sock != null) sock.close();
            } catch (Exception ignored) {
            }
            sock = null;
        }
    }
}
