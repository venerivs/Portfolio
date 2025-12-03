
// Andrew Wineinger
// RP2040w/RFM - Transmitter

#include <SPI.h>
#include <RH_RF95.h>

#define RFM95_CS   16
#define RFM95_INT  21
#define RFM95_RST  17

// Change to 434.0 or other frequency, must match RX's freq!
#define RF95_FREQ 915.0

// Singleton instance of the radio driver
RH_RF95 rf95(RFM95_CS, RFM95_INT);

void setup() {
  pinMode(RFM95_RST, OUTPUT);
  digitalWrite(RFM95_RST, HIGH);

  Serial.begin(115200);
  while (!Serial) delay(1);
  delay(100);

  Serial.println("Feather LoRa TX Test!");

  // Manual reset
  digitalWrite(RFM95_RST, LOW);
  delay(10);
  digitalWrite(RFM95_RST, HIGH);
  delay(10);

  // Initialize LoRa radio
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

  // Set maximum preamble length (65535 symbols)
  rf95.spiWrite(0x20, 0xFF);  // Set Preamble length MSB
  rf95.spiWrite(0x21, 0xFF);  // Set Preamble length LSB
  Serial.println("Preamble length set to max (65535 symbols)");

  // Set Tx power to maximum (23 dBm)
  rf95.setTxPower(23, false);
  Serial.println("Transmitter power set to 23 dBm");

  // Verify preamble length
  uint8_t preamble_msb = rf95.spiRead(0x20);
  uint8_t preamble_lsb = rf95.spiRead(0x21);
  uint16_t preamble_length = (preamble_msb << 8) | preamble_lsb;

  Serial.print("Verified Preamble Length: ");
  Serial.println(preamble_length);
}

int16_t packetnum = 0;  // packet counter, we increment per transmission


void loop() {
  delay(1000); // Wait 1 second between transmits, could also 'sleep' here!
  Serial.println("Transmitting..."); // Send a message to rf95_server

  char radiopacket[20] = "Hello World #      ";
  itoa(packetnum++, radiopacket+13, 10);
  Serial.print("Sending "); Serial.println(radiopacket);
  radiopacket[19] = 0;

  Serial.println("Sending...");
  delay(10);
  rf95.send((uint8_t *)radiopacket, 20);

  Serial.println("Waiting for packet to complete...");
  delay(10);
  rf95.waitPacketSent();
  // Now wait for a reply


}