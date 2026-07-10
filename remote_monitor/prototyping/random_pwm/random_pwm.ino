/*
 * Random PWM Signal Generator
 * ============================
 * Outputs a PWM signal on PWM_PIN with:
 *   - Random duty cycle : 10 – 90 %
 *   - Random ON duration: 3 – 15 seconds
 *   - Random OFF interval between signals: 5 – 30 seconds
 *
 * Compatible with Uno, Nano, Mega, and most AVR-based boards.
 * Change PWM_PIN to any PWM-capable pin for your board.
 */

// ── Configuration ────────────────────────────────────────────────────────────
const uint8_t PWM_PIN = 9;       // Must be a PWM-capable pin (3,5,6,9,10,11 on Uno/Nano)

// Duty cycle range (%)
const uint8_t DC_MIN  = 10;
const uint8_t DC_MAX  = 90;

// Active (ON) duration range (seconds)
const uint8_t ON_MIN  = 3;
const uint8_t ON_MAX  = 15;

// Idle (OFF) interval range (seconds)
const uint8_t OFF_MIN = 5;
const uint8_t OFF_MAX = 30;
// ── End Configuration ────────────────────────────────────────────────────────


// Helper: return a random integer in [lo, hi] (inclusive)
static uint8_t randRange(uint8_t lo, uint8_t hi) {
  return lo + (uint8_t)(random(hi - lo + 1));
}

void setup() {
  pinMode(PWM_PIN, OUTPUT);
  analogWrite(PWM_PIN, 0);          // Start with pin LOW

  // Seed the PRNG from a floating analog pin for true randomness
  randomSeed(analogRead(A0));

  Serial.begin(9600);
  Serial.println(F("Random PWM Generator started"));
  Serial.println(F("Pin | Duty% | ON(s) | OFF(s)"));
  Serial.println(F("----+-------+-------+-------"));
}

void loop() {
  // ── Pick random parameters for this burst ───────────────────────────────
  uint8_t dutyCycle  = randRange(DC_MIN,  DC_MAX);   // 10 – 90 %
  uint8_t onSeconds  = randRange(ON_MIN,  ON_MAX);   //  3 – 15 s
  uint8_t offSeconds = randRange(OFF_MIN, OFF_MAX);  //  5 – 30 s

  // Convert duty cycle % → 8-bit PWM value (0–255)
  uint8_t pwmValue = map(dutyCycle, 0, 100, 0, 255);

  // ── Log parameters to Serial Monitor ───────────────────────────────────
  Serial.print(F("D"));
  Serial.print(PWM_PIN);
  Serial.print(F("  | "));
  Serial.print(dutyCycle);
  Serial.print(F("%   | "));
  Serial.print(onSeconds);
  Serial.print(F("s     | "));
  Serial.print(offSeconds);
  Serial.println(F("s"));

  // ── PWM ON phase ────────────────────────────────────────────────────────
  Serial.println("PWM >>> ON");
  analogWrite(PWM_PIN, pwmValue);
  delay((uint32_t)onSeconds * 1000UL);

  // ── PWM OFF phase ───────────────────────────────────────────────────────
  Serial.println("PWM >>> OFF");
  analogWrite(PWM_PIN, 0);
  delay((uint32_t)offSeconds * 1000UL);
}
