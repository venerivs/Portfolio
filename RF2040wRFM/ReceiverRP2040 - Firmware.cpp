  // Andrew Wineinger
  // RP2040/RFM - Receiver Code
  // mode: C++

  #include <SPI.h>
  #include <RH_RF95.h>

  #define LORA_FREQUENCY     915000000  // 915 MHz
  #define LORA_BANDWIDTH     0          // 125 kHz
  #define LORA_SPREADINGFACTOR 7        // SF7
  #define LORA_CODINGRATE    1          // 4/5
  #define LORA_TX_POWER      22         // Max 22 dBm for SX1260
  #define LORA_CRC_ENABLED   1          // Enable CRC

  //RP2040 w/ RFM
  #define RFM95_CS   16
  #define RFM95_INT  21
  #define RFM95_RST  17

  // Change to 434.0 or other frequency, must match RX's freq!
  #define RF95_FREQ 915.0

  // Singleton instance of the radio driver
  RH_RF95 rf95(RFM95_CS, RFM95_INT);

  void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    pinMode(RFM95_RST, OUTPUT);
    digitalWrite(RFM95_RST, HIGH);

    Serial.begin(115200);
    while (!Serial) delay(1);
    delay(100);

    Serial.println("Feather LoRa RX Test!");

    // Manual reset of RFM95 module
    digitalWrite(RFM95_RST, LOW);
    delay(10);
    digitalWrite(RFM95_RST, HIGH);
    delay(10);

    // Initialize LoRa module
    while (!rf95.init()) {
        Serial.println("LoRa radio init failed");
        Serial.println("Uncomment '#define SERIAL_DEBUG' in RH_RF95.cpp for detailed debug info");
        while (1);
    }
    Serial.println("LoRa radio init OK!");

    // Set frequency
    if (!rf95.setFrequency(RF95_FREQ)) {
        Serial.println("setFrequency failed");
        while (1);
    }
    Serial.print("Set Freq to: "); Serial.println(RF95_FREQ);

    // Set Tx power to max (23 dBm)
    rf95.setTxPower(23, false);
    Serial.println("Transmitter power set to 23 dBm");

    // Set maximum preamble length (0xFFFF = 65535 symbols)
    rf95.spiWrite(0x20, 0xFF);  // Set Preamble length MSB
    rf95.spiWrite(0x21, 0xFF);  // Set Preamble length LSB
    Serial.println("Preamble length set to max (65535 symbols)");

    // Verify preamble length
    uint8_t preamble_msb = rf95.spiRead(0x20);
    uint8_t preamble_lsb = rf95.spiRead(0x21);
    uint16_t preamble_length = (preamble_msb << 8) | preamble_lsb;

    Serial.print("Verified Preamble Length: ");
    Serial.println(preamble_length);
}


  void loop() {
    if (rf95.available()) {
      // Send out looped messages
      uint8_t buf[RH_RF95_MAX_MESSAGE_LEN];
      uint8_t len = sizeof(buf);

      if (rf95.recv(buf, &len)) {
        digitalWrite(LED_BUILTIN, HIGH);
        RH_RF95::printBuffer("Received: ", buf, len);
        Serial.print("Got: ");
        Serial.println((char*)buf);
        Serial.print("RSSI: ");
        Serial.println(rf95.lastRssi(), DEC);

      } else {
        Serial.println("Receive failed");
      }
    }
  }