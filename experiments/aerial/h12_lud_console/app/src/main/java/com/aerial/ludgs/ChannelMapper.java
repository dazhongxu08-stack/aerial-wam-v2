package com.aerial.ludgs;

/** H12 PWM (≈1000–2000) → Lumos walk speeds. MODE 2. */
public final class ChannelMapper {
    public static final int CENTER = 1500;
    public static final int MIN = 1000;
    public static final int MAX = 2000;
    public static final int DEADBAND = 40;
    public static final int SWITCH_LOW_MAX = 1300;
    public static final int SWITCH_HIGH_MIN = 1700;

    public static final int STATE_RESET = 1;
    public static final int STATE_STAND = 2;
    public static final int STATE_LIEDOWN = 5;
    public static final int STATE_WALK = 12;

    public static final float VX_MAX = 0.5f;
    public static final float VY_MAX = 0.3f;
    public static final float VYAW_MAX = 0.5f;

    public enum TriPos {
        LOW, MID, HIGH, UNKNOWN
    }

    public static final class Cmd {
        public int state = STATE_WALK;
        public float vx;
        public float vy;
        public float vyaw;
        public boolean stop;
    }

    private ChannelMapper() {}

    public static int pwmToRc(int pwm) {
        if (pwm <= 0) {
            return 0;
        }
        int v = Math.max(MIN, Math.min(MAX, pwm));
        int delta = v - CENTER;
        if (Math.abs(delta) < DEADBAND) {
            return 0;
        }
        float signed = delta > 0
                ? (delta - DEADBAND) / (float) (MAX - CENTER - DEADBAND)
                : (delta + DEADBAND) / (float) (CENTER - MIN - DEADBAND);
        int out = Math.round(signed * 100f);
        return Math.max(-100, Math.min(100, out));
    }

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
     * ch1 roll → vy (inverted), ch2 pitch → vx (inverted), ch4 yaw → vyaw (inverted).
     */
    public static Cmd toLumos(int[] ch, int holdState, boolean outputOn) {
        Cmd cmd = new Cmd();
        cmd.state = holdState;
        if (!outputOn || ch == null) {
            return cmd;
        }
        int roll = ch.length > 0 ? -pwmToRc(ch[0]) : 0;
        int pitch = ch.length > 1 ? -pwmToRc(ch[1]) : 0;
        int yaw = ch.length > 3 ? -pwmToRc(ch[3]) : 0;
        if (ch.length > 4 && triPos(ch[4]) == TriPos.HIGH) {
            cmd.stop = true;
            return cmd;
        }
        if (holdState != STATE_WALK) {
            return cmd;
        }
        cmd.vx = clamp(pitch / 100f * VX_MAX, -VX_MAX, VX_MAX);
        cmd.vy = clamp(roll / 100f * VY_MAX, -VY_MAX, VY_MAX);
        cmd.vyaw = clamp(yaw / 100f * VYAW_MAX, -VYAW_MAX, VYAW_MAX);
        return cmd;
    }

    public static int stateFromCh6(TriPos z6, int current) {
        if (z6 == TriPos.HIGH) {
            return STATE_WALK;
        }
        if (z6 == TriPos.MID) {
            return STATE_STAND;
        }
        if (z6 == TriPos.LOW) {
            return STATE_LIEDOWN;
        }
        return current;
    }

    /** CH5 mid = RESET; otherwise CH6 owns gait. */
    public static int stateFromSwitches(TriPos z5, TriPos z6, int current) {
        if (z5 == TriPos.MID) {
            return STATE_RESET;
        }
        return stateFromCh6(z6, current);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
