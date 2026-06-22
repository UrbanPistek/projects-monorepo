/*
  
*/

volatile int pulseCount = 0;
float flowRate = 0.0;
unsigned long lastTime = 0;

// This factor varies by sensor — check your datasheet
// Common value: 5.5 pulses/sec = 1 L/min
const float calibrationFactor = 5.5;

void pulseCounter() {
  pulseCount++;
}

// the setup function runs once when you press reset or power the board
void setup() {
  
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);

  pinMode(2, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(2), pulseCounter, FALLING);
  
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);

  Serial.println("> Setup Complete");
}

// the loop function runs over and over again forever
void loop() {

  if (millis() - lastTime >= 1000) {  // Calculate every second
    detachInterrupt(digitalPinToInterrupt(2));

    flowRate = (pulseCount / calibrationFactor);  // Litres per minute

    Serial.print("Flow rate: ");
    Serial.print(flowRate);
    Serial.println(" L/min");

    pulseCount = 0;
    lastTime = millis();

    attachInterrupt(digitalPinToInterrupt(2), pulseCounter, FALLING);
  }  
  
}
