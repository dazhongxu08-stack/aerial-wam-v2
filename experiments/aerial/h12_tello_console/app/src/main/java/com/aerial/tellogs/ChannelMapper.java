package com.aerial.tellogs;

/** Map H12 PWM channels (≈1000–2000) → Tello SDK RC (-100..100). Mode 2 / US. */
public final class ChannelMapper {
    public static final int CENTER = 1500;
    public static final int MIN = 1000;
    public static final int MAX = 2000;
    public static final int DEADBAND = 40;
    /** 三段拨杆：低 ≤1300，中 1301–1699，高 ≥1700。 */
    public static final int SWITCH_LOW_MAX = 1300;
    public static final int SWITCH_HIGH_MIN = 1700;

    public enum TriPos {
        LOW,
        MID,
        HIGH,
        UNKNOWN
    }

    private ChannelMapper() {}

    /** PWM → -100..100 with deadband around center. */
    public static int pwmToRc(int pwm) {
        if (pwm <= 0) {
            return 0;
        }
        int v = Math.max(MIN, Math.min(MAX, pwm));
        int delta = v - CENTER;
        if (Math.abs(delta) < DEADBAND) {
            return 0;
        }
        // scale remaining range
        float signed = delta > 0
                ? (delta - DEADBAND) / (float) (MAX - CENTER - DEADBAND)
                : (delta + DEADBAND) / (float) (CENTER - MIN - DEADBAND);
        int out = Math.round(signed * 100f);
        return Math.max(-100, Math.min(100, out));
    }

    /** 三段开关分区（ch5/ch6 等拨杆）。 */
    public static TriPos triPos(int pwm) {
        if (pwm <= 0) {
            return TriPos.UNKNOWN;
        }
        if (pwm <= SWITCH_LOW_MAX) {
            return TriPos.LOW;
        }
        if (pwm >= SWITCH_HIGH_MIN) {
            return TriPos.HIGH;
        }
        return TriPos.MID;
    }

    /**
     * @param ch 1-based channel array from RCSDK (index 0 = ch1)
     * @return int[4] = {a roll, b pitch, c throttle, d yaw}
     *         Tello: +b = forward. H12 MODE2 pitch stick is inverted vs Tello, so negate ch2.
     */
    public static int[] toTelloRc(int[] ch) {
        int a = 0, b = 0, c = 0, d = 0;
        if (ch != null) {
            if (ch.length > 0) a = pwmToRc(ch[0]); // ch1 roll
            if (ch.length > 1) b = -pwmToRc(ch[1]); // ch2 pitch (H12 fwd/back reversed)
            if (ch.length > 2) c = pwmToRc(ch[2]); // ch3 throttle
            if (ch.length > 3) d = pwmToRc(ch[3]); // ch4 yaw
        }
        return new int[]{a, b, c, d};
    }
}
