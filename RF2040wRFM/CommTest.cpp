void setup() {
    Serial.begin(9600);  // Initialize serial communication at 9600 baud
    randomSeed(analogRead(0));  // Seed the random number generator
}

void loop() {
    float x = 30.0 + static_cast<float>(random(0, 5000)) / 10000.0;  // Generate a random x value between 30 and 31
    float y = -84.0 + static_cast<float>(random(0, 5000)) / 10000.0;  // Generate a random y value between 84 and 85
    int randomNum = random(0, 101);  // Generate a random number between 0 and 100
    
    Serial.print(x, 4);
    Serial.print(",");
    Serial.print(y, 4);
    Serial.print(",");
    Serial.println(randomNum);
    
    delay(5000);  // Wait for 5 seconds before generating new numbers
}
