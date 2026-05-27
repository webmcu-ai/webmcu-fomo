// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// FULL VISION ML WITH FOMO HEAD — WebSerial Edition — v016
//
// Derived from on-device-fomo v015 by Jeremy Ellis
// WebSerial hooks adapted from webmcu-vision-web v70
//
// Suggested repo: webmcu-fomo-web
//
// WHAT THIS FIRMWARE ADDS OVER THE ON-DEVICE v015:
//   - ArduinoJson config system (MyConfig / config.json) — dynamic class counts
//   - WebSerial full string-command parser (myHandleStringCommand)
//   - SD browser: SD_LIST, SD_READ, SD_WRITE, SD_DELETE, SD_RMDIR, SD_MKDIR
//   - SD chunked JPEG write: SD_JPEG_WRITE_START / SD_JPEG_CHUNK / SD_JPEG_WRITE_END
//   - SD chunked text write: SD_TEXT_WRITE_START / SD_TEXT_CHUNK / SD_TEXT_WRITE_END
//   - Binary weight transfer: FILE_SEND_START / FILE_CHUNK / FILE_SEND_END  (v78 protocol)
//   - Camera stream: CAM_CAPTURE, CAM_STREAM, CAM_STREAM_STOP
//   - Heatmap streaming: HEATMAP_ON, HEATMAP_OFF, HEATMAP_STATUS
//   - FOMO detection threshold command: DETECTION_THRESHOLD:<value>
//   - TRAIN_EPOCH, COLLECT_COUNT, INFER serial tokens for webpage sync
//   - Graceful SD-absent boot; inference of baked/RAM weights without SD
//   - esp_log spam suppression (FB_OVF silenced)
//
// INFER serial format (sent after every myForwardPass in inference):
//   INFER:<className>,<gapConf>,<peakX>,<peakY>,<peakVal>,<numClusters>
//   Example: INFER:1Cup,0.4231,14,9,0.8750,2
//
// COLLECT_COUNT serial format (sent after every image capture):
//   COLLECT_COUNT:<classIdx>,<count>
//
// TRAIN_EPOCH serial format (sent at end of every epoch):
//   TRAIN_EPOCH:<epoch>,<loss>,<accuracy>
//
// MIT license — Jeremy Ellis  https://github.com/hpssjellis
//
// For Arduino IDE:
//   Board: XIAO_ESP32S3 (Seeed Studio XIAO ESP32S3 Sense)
//   PSRAM: OPI PSRAM
//   USB CDC On Boot: Enabled
//   Flash Size: 8MB (64Mb)
//   Flash Mode: QIO 80MHz
//
// Libraries (Sketch -> Include Library -> Manage Libraries):
//   U8g2 by olikraus
//   ArduinoJson by Benoit Blanchon >= 7.x
//
// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 0: CORE SYSTEM (ALWAYS INCLUDED)                                   ██
// ██  Includes, Config, Globals, Touch, Setup, Loop                           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// Optional: uncomment AFTER copying myWeights.h from SD to your sketch folder.
// Priority order: SD weights > baked-in weights > random He-init
//#define USE_BAKED_WEIGHTS
#ifdef USE_BAKED_WEIGHTS
  #include "myWeights.h"
#endif

#include "esp_camera.h"
#include "img_converters.h"
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include "esp_log.h"
#include "mbedtls/base64.h"   // built into ESP32 SDK — no install needed

U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);  // 180° re-orientation

// ======================================================
// COMPILED-IN DEFAULTS  (FOMO-specific)
// Used when /header/config.json is absent or a key is missing.
// Written out as config.json on first boot.
// ======================================================
#define DEFAULT_INPUT_SIZE        64
#define DEFAULT_NUM_CLASSES        2
#define DEFAULT_CONV1_FILTERS      4
#define DEFAULT_CONV2_FILTERS      8
#define DEFAULT_LEARNING_RATE      0.0003f
#define DEFAULT_BATCH_SIZE        12
#define DEFAULT_TARGET_EPOCHS     30
#define DEFAULT_THRESHOLD_PRESS  1100
#define DEFAULT_THRESHOLD_RELEASE 900
#define DEFAULT_SCREEN_TIMEOUT    300000UL   // 5 minutes in ms
#define DEFAULT_WEIGHTS_FILE      "myWeights.bin"

// Versioning — monotonically increasing integer.
// MIN_VERSION = oldest config.json / .bin accepted without halting.
#define CURRENT_VERSION 16
#define MIN_VERSION     16

// FOMO-specific defaults
#define DEFAULT_VALIDATION_IMAGES    5
#define DEFAULT_USE_AUGMENTATION     false
#define DEFAULT_FOMO_THRESHOLD       0.38f
#define DEFAULT_USE_DYNAMIC_THRESH   true
#define DEFAULT_DYNAMIC_THRESH_RATIO 0.90f
#define DEFAULT_DYNAMIC_THRESH_FLOOR 0.38f

#define DEFAULT_NUM_LABELS 2
const char* DEFAULT_LABELS[] = { "0Blank", "1Cup" };

// ======================================================
// RUNTIME CONFIG STRUCT
// ======================================================
struct MyConfig {
  int   inputSize;
  int   numClasses;
  int   conv1Filters;
  int   conv2Filters;

  // Derived sizes (computed by myComputeArchSizes)
  int conv1OutputSize;
  int pool1OutputSize;
  int conv2OutputSize;
  int flattenedSize;     // = conv2OutputSize^2 * conv2Filters
  int fomoGrid;          // = conv2OutputSize
  int fomoCells;         // = fomoGrid^2
  int conv1Weights;
  int conv2Weights;
  int outputWeights;     // = flattenedSize * numClasses  (kept for weight compat)

  float  learningRate;
  int    batchSize;
  int    targetEpochs;
  int    thresholdPress;
  int    thresholdRelease;
  unsigned long screenTimeout;
  int    numLabels;
  String classLabels[8];
  String weightsFile;

  int version;
  int minVersion;

  // Training options
  bool  useAugmentation;
  int   validationImages;

  // FOMO detection options
  float fomoThreshold;
  bool  useDynamicThreshold;
  float dynamicThresholdRatio;
  float dynamicThresholdFloor;
};

MyConfig myCfg;

void myComputeArchSizes() {
  myCfg.conv1OutputSize = myCfg.inputSize - 2;
  myCfg.pool1OutputSize = myCfg.conv1OutputSize / 2;
  myCfg.conv2OutputSize = myCfg.pool1OutputSize - 2;
  myCfg.flattenedSize   = myCfg.conv2OutputSize * myCfg.conv2OutputSize * myCfg.conv2Filters;
  myCfg.fomoGrid        = myCfg.conv2OutputSize;
  myCfg.fomoCells       = myCfg.fomoGrid * myCfg.fomoGrid;
  myCfg.conv1Weights    = 3 * 3 * 3 * myCfg.conv1Filters;           // always RGB (3 channels)
  myCfg.conv2Weights    = 3 * 3 * myCfg.conv1Filters * myCfg.conv2Filters;
  myCfg.outputWeights   = myCfg.flattenedSize * myCfg.numClasses;   // kept for .bin compat
}

void myApplyDefaultConfig() {
  myCfg.inputSize        = DEFAULT_INPUT_SIZE;
  myCfg.numClasses       = DEFAULT_NUM_CLASSES;
  myCfg.conv1Filters     = DEFAULT_CONV1_FILTERS;
  myCfg.conv2Filters     = DEFAULT_CONV2_FILTERS;
  myCfg.learningRate     = DEFAULT_LEARNING_RATE;
  myCfg.batchSize        = DEFAULT_BATCH_SIZE;
  myCfg.targetEpochs     = DEFAULT_TARGET_EPOCHS;
  myCfg.thresholdPress   = DEFAULT_THRESHOLD_PRESS;
  myCfg.thresholdRelease = DEFAULT_THRESHOLD_RELEASE;
  myCfg.screenTimeout    = DEFAULT_SCREEN_TIMEOUT;
  myCfg.weightsFile      = DEFAULT_WEIGHTS_FILE;
  myCfg.numLabels        = DEFAULT_NUM_LABELS;
  for (int i = 0; i < DEFAULT_NUM_LABELS; i++)
    myCfg.classLabels[i] = String(DEFAULT_LABELS[i]);
  myCfg.version             = CURRENT_VERSION;
  myCfg.minVersion          = MIN_VERSION;
  myCfg.useAugmentation     = DEFAULT_USE_AUGMENTATION;
  myCfg.validationImages    = DEFAULT_VALIDATION_IMAGES;
  myCfg.fomoThreshold       = DEFAULT_FOMO_THRESHOLD;
  myCfg.useDynamicThreshold = DEFAULT_USE_DYNAMIC_THRESH;
  myCfg.dynamicThresholdRatio = DEFAULT_DYNAMIC_THRESH_RATIO;
  myCfg.dynamicThresholdFloor = DEFAULT_DYNAMIC_THRESH_FLOOR;
  myComputeArchSizes();
}

void mySaveConfig() {
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open("/header/config.json", FILE_WRITE);
  if (!f) { Serial.println("[config] ERROR: could not write config.json"); return; }
  JsonDocument doc;
  doc["inputSize"]             = myCfg.inputSize;
  doc["numClasses"]            = myCfg.numClasses;
  doc["conv1Filters"]          = myCfg.conv1Filters;
  doc["conv2Filters"]          = myCfg.conv2Filters;
  doc["learningRate"]          = myCfg.learningRate;
  doc["batchSize"]             = myCfg.batchSize;
  doc["targetEpochs"]          = myCfg.targetEpochs;
  doc["thresholdPress"]        = myCfg.thresholdPress;
  doc["thresholdRelease"]      = myCfg.thresholdRelease;
  doc["screenTimeout"]         = (unsigned long)myCfg.screenTimeout;
  doc["weightsFile"]           = myCfg.weightsFile;
  doc["version"]               = myCfg.version;
  doc["minVersion"]            = myCfg.minVersion;
  doc["useAugmentation"]       = myCfg.useAugmentation;
  doc["validationImages"]      = myCfg.validationImages;
  doc["fomoThreshold"]         = myCfg.fomoThreshold;
  doc["useDynamicThreshold"]   = myCfg.useDynamicThreshold;
  doc["dynamicThresholdRatio"] = myCfg.dynamicThresholdRatio;
  doc["dynamicThresholdFloor"] = myCfg.dynamicThresholdFloor;
  JsonArray labels = doc["classLabels"].to<JsonArray>();
  for (int i = 0; i < myCfg.numLabels; i++) labels.add(myCfg.classLabels[i]);
  serializeJsonPretty(doc, f);
  f.close();
  Serial.println("[config] Wrote config to /header/config.json");
}

void myLoadConfig() {
  myApplyDefaultConfig();
  const char* path = "/header/config.json";
  if (!SD.exists(path)) {
    Serial.println("[config] config.json not found — using defaults");
    mySaveConfig(); return;
  }
  File f = SD.open(path, FILE_READ);
  if (!f) { Serial.println("[config] ERROR: could not open config.json"); return; }
  JsonDocument doc;
  DeserializationError err = deserializeJson(doc, f);
  f.close();
  if (err) {
    Serial.printf("[config] JSON parse error: %s — using defaults\n", err.c_str()); return;
  }
  myCfg.inputSize             = doc["inputSize"]             | DEFAULT_INPUT_SIZE;
  myCfg.numClasses            = doc["numClasses"]            | DEFAULT_NUM_CLASSES;
  myCfg.conv1Filters          = doc["conv1Filters"]          | DEFAULT_CONV1_FILTERS;
  myCfg.conv2Filters          = doc["conv2Filters"]          | DEFAULT_CONV2_FILTERS;
  myCfg.learningRate          = doc["learningRate"]          | DEFAULT_LEARNING_RATE;
  myCfg.batchSize             = doc["batchSize"]             | DEFAULT_BATCH_SIZE;
  myCfg.targetEpochs          = doc["targetEpochs"]          | DEFAULT_TARGET_EPOCHS;
  myCfg.thresholdPress        = doc["thresholdPress"]        | DEFAULT_THRESHOLD_PRESS;
  myCfg.thresholdRelease      = doc["thresholdRelease"]      | DEFAULT_THRESHOLD_RELEASE;
  myCfg.screenTimeout         = doc["screenTimeout"]         | (int)DEFAULT_SCREEN_TIMEOUT;
  myCfg.weightsFile           = doc["weightsFile"]           | DEFAULT_WEIGHTS_FILE;
  myCfg.version               = doc["version"]               | CURRENT_VERSION;
  myCfg.minVersion            = doc["minVersion"]            | MIN_VERSION;
  myCfg.useAugmentation       = doc["useAugmentation"]       | DEFAULT_USE_AUGMENTATION;
  myCfg.validationImages      = doc["validationImages"]      | DEFAULT_VALIDATION_IMAGES;
  myCfg.fomoThreshold         = doc["fomoThreshold"]         | DEFAULT_FOMO_THRESHOLD;
  myCfg.useDynamicThreshold   = doc["useDynamicThreshold"]   | DEFAULT_USE_DYNAMIC_THRESH;
  myCfg.dynamicThresholdRatio = doc["dynamicThresholdRatio"] | DEFAULT_DYNAMIC_THRESH_RATIO;
  myCfg.dynamicThresholdFloor = doc["dynamicThresholdFloor"] | DEFAULT_DYNAMIC_THRESH_FLOOR;

  if (myCfg.version < MIN_VERSION) {
    Serial.printf("[config] ERROR: config.json version %d < minimum %d\n", myCfg.version, MIN_VERSION);
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 10, "CONFIG ERROR");
      u8g2.drawStr(0, 22, "Old version!");
    } while (u8g2.nextPage());
    while (true) { delay(1000); }
  }

  if (doc["classLabels"].is<JsonArray>()) {
    JsonArray arr = doc["classLabels"].as<JsonArray>();
    int count = 0;
    for (JsonVariant v : arr) {
      if (count >= 8) break;
      myCfg.classLabels[count++] = v.as<String>();
    }
    if (count > 0) { myCfg.numLabels = count; myCfg.numClasses = count; }
  }

  myComputeArchSizes();
  Serial.println("[config] Loaded /header/config.json:");
  Serial.printf("  inputSize=%d numClasses=%d conv1F=%d conv2F=%d\n",
                myCfg.inputSize, myCfg.numClasses, myCfg.conv1Filters, myCfg.conv2Filters);
  Serial.printf("  fomoGrid=%d fomoCells=%d\n", myCfg.fomoGrid, myCfg.fomoCells);
  Serial.printf("  learningRate=%.5f batchSize=%d targetEpochs=%d\n",
                myCfg.learningRate, myCfg.batchSize, myCfg.targetEpochs);
  Serial.printf("  fomoThreshold=%.3f useDynamic=%s ratio=%.2f floor=%.2f\n",
                myCfg.fomoThreshold, myCfg.useDynamicThreshold ? "true" : "false",
                myCfg.dynamicThresholdRatio, myCfg.dynamicThresholdFloor);
}

// ======================================================
// LEGACY SHIM — keeps myClassLabels[] usable in Parts 1-4
// ======================================================
String myClassLabels[8];
void mySyncLegacyVars() {
  for (int i = 0; i < myCfg.numLabels; i++)
    myClassLabels[i] = myCfg.classLabels[i];
}

// Convenience macros that read from myCfg at runtime
#define LEARNING_RATE    (myCfg.learningRate)
#define BATCH_SIZE       (myCfg.batchSize)
#define TARGET_EPOCHS    (myCfg.targetEpochs)
#define NUM_CLASSES      (myCfg.numClasses)
#define INPUT_SIZE       (myCfg.inputSize)
#define CONV1_FILTERS    (myCfg.conv1Filters)
#define CONV2_FILTERS    (myCfg.conv2Filters)
#define CONV1_WEIGHTS    (myCfg.conv1Weights)
#define CONV2_WEIGHTS    (myCfg.conv2Weights)
#define OUTPUT_WEIGHTS   (myCfg.outputWeights)
#define FLATTENED_SIZE   (myCfg.flattenedSize)
#define CONV1_OUTPUT_SIZE (myCfg.conv1OutputSize)
#define POOL1_OUTPUT_SIZE (myCfg.pool1OutputSize)
#define CONV2_OUTPUT_SIZE (myCfg.conv2OutputSize)
#define FOMO_GRID        (myCfg.fomoGrid)
#define FOMO_CELLS       (myCfg.fomoCells)

// v76: dynamic menu dispatch — myTotalItems = NUM_CLASSES + 2 (Train + Infer)
#define myTotalItems (myCfg.numClasses + 2)

#define myThresholdPress   (myCfg.thresholdPress)
#define myThresholdRelease (myCfg.thresholdRelease)

// ======================================================
// XIAO ESP32-S3 CAMERA PINS
// ======================================================
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    10
#define SIOD_GPIO_NUM    40
#define SIOC_GPIO_NUM    39
#define Y9_GPIO_NUM      48
#define Y8_GPIO_NUM      11
#define Y7_GPIO_NUM      12
#define Y6_GPIO_NUM      14
#define Y5_GPIO_NUM      16
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM      17
#define Y2_GPIO_NUM      15
#define VSYNC_GPIO_NUM   38
#define HREF_GPIO_NUM    47
#define PCLK_GPIO_NUM    13

// ======================================================
// SYSTEM LOGIC VARIABLES
// ======================================================
bool mySDavailable    = false;
bool myWeightsTrained = false;
bool myHeatmapEnabled = false;

unsigned long myLastActivityTime = 0;
unsigned long myLastTapTime      = 0;
const int     myTapCooldown      = 250;
int           myMenuIndex        = 1;
bool          myIsSelected       = false;

// Adam optimizer step counter — persists across warm retrains
int myAdamStep = 0;

// ======================================================
// UNIFIED TOUCH INPUT SYSTEM
// ======================================================
struct TouchState {
  bool isTouching       = false;
  int  tapCount         = 0;
  unsigned long firstTapTime    = 0;
  unsigned long lastReleaseTime = 0;
  unsigned long lastCheckTime   = 0;
  const unsigned long tapWindow    = 800;
  const int           longPressTaps = 3;
  const unsigned long debounceDelay = 50;
};
TouchState myTouch;

// ======================================================
// WEBSERIAL STRING COMMAND STATE
// ======================================================
String mySerialLineBuf = "";
bool   myCamStreaming  = false;
unsigned long myLastCamStreamMs = 0;
const uint16_t MY_CAM_STREAM_INTERVAL_MS = 200;  // ~5 fps

// Chunked JPEG write state (SD_JPEG_WRITE_START / CHUNK / END)
String myJpegWritePath   = "";
String myJpegWriteB64    = "";
bool   myJpegWriteActive = false;

// Chunked text write state (SD_TEXT_WRITE_START / CHUNK / END)
String myTextWritePath    = "";
String myTextWriteContent = "";
bool   myTextWriteActive  = false;

// Binary file receive state (FILE_SEND_START / FILE_CHUNK / FILE_SEND_END) — v78 protocol
String myFileRecvPath    = "";
File   myFileRecvHandle;
int    myFileRecvBytes   = 0;
bool   myFileRecvActive  = false;

// ======================================================
// GLOBAL ML BUFFERS (all PSRAM)
// ======================================================
uint8_t* myRgbBuffer = nullptr;

float* myInputBuffer  = nullptr;
float* myConv1_w      = nullptr;
float* myConv1_b      = nullptr;
float* myConv2_w      = nullptr;
float* myConv2_b      = nullptr;
float* myOutput_w     = nullptr;   // 1x1 conv weights [numClasses][conv2Filters]
float* myOutput_b     = nullptr;   // 1x1 conv biases  [numClasses]

float* myConv1_w_grad  = nullptr;
float* myConv1_b_grad  = nullptr;
float* myConv2_w_grad  = nullptr;
float* myConv2_b_grad  = nullptr;
float* myOutput_w_grad = nullptr;
float* myOutput_b_grad = nullptr;

float* myConv1_w_m  = nullptr;  float* myConv1_w_v  = nullptr;
float* myConv1_b_m  = nullptr;  float* myConv1_b_v  = nullptr;
float* myConv2_w_m  = nullptr;  float* myConv2_w_v  = nullptr;
float* myConv2_b_m  = nullptr;  float* myConv2_b_v  = nullptr;
float* myOutput_w_m = nullptr;  float* myOutput_w_v = nullptr;
float* myOutput_b_m = nullptr;  float* myOutput_b_v = nullptr;

float* myConv1_output = nullptr;
float* myPool1_output = nullptr;
float* myConv2_output = nullptr;

// FOMO map: [numClasses * fomoCells] — sigmoid output per class per cell
float* myFomoMap     = nullptr;
// Global average pool of FOMO map — used for predicted class and validation
float* myDense_output = nullptr;

// Gradient propagation buffers
float* myFomoGrad   = nullptr;   // grad w.r.t. conv2_output: [fomoCells * conv2Filters]
float* myConv2_grad = nullptr;
float* myPool1_grad = nullptr;
float* myConv1_grad = nullptr;

// ======================================================
// BOUNDING BOX / TRAINING ITEM TYPES
// ======================================================
struct FomoBox { float x1, y1, x2, y2; };
const FomoBox myDefaultBox = { 0.35f, 0.35f, 0.65f, 0.65f };

struct TrainingItem {
  String path;
  int    label;
  std::vector<FomoBox> boxes;
};
std::vector<TrainingItem> myTrainingData;

// ======================================================
// UTILITY FUNCTIONS
// ======================================================
inline float clip_value(float v, float mn = -100, float mx = 100) {
  if (isnan(v) || isinf(v)) return 0;
  return constrain(v, mn, mx);
}
inline float leaky_relu(float x)       { return x > 0 ? x : 0.1f * x; }
inline float leaky_relu_deriv(float x) { return x > 0 ? 1.0f : 0.1f; }
inline float my_sigmoid(float x)       { return 1.0f / (1.0f + expf(-x)); }
inline float my_sigmoid_deriv(float s) { return s * (1.0f - s); }   // s = sigmoid(x)

// Flip myRgbBuffer (240x240 RGB888) horizontally in-place
void myFlipImageHorizontal() {
  for (int y = 0; y < 240; y++) {
    uint8_t* row = myRgbBuffer + y * 240 * 3;
    for (int x = 0; x < 120; x++) {
      uint8_t* left  = row + x * 3;
      uint8_t* right = row + (239 - x) * 3;
      uint8_t tmp;
      tmp = left[0]; left[0] = right[0]; right[0] = tmp;
      tmp = left[1]; left[1] = right[1]; right[1] = tmp;
      tmp = left[2]; left[2] = right[2]; right[2] = tmp;
    }
  }
}

// ======================================================
// UNIFIED TOUCH INPUT FUNCTIONS
// ======================================================
int myReadTouch() {
  int sum = 0;
  for (int i = 0; i < 3; i++) { sum += analogRead(A0); delayMicroseconds(100); }
  return sum / 3;
}
void myResetTouchState() {
  myTouch.isTouching = false;
  myTouch.tapCount   = 0;
  myTouch.firstTapTime = myTouch.lastReleaseTime = myTouch.lastCheckTime = 0;
}
void myUpdateTouchState() {
  unsigned long now = millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool touchActive = myTouch.isTouching ? (val > myThresholdRelease) : (val > myThresholdPress);
  if (touchActive && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount == 0 || (now - myTouch.firstTapTime < myTouch.tapWindow)) {
      if (myTouch.tapCount == 0) myTouch.firstTapTime = now;
      myTouch.tapCount++;
      Serial.printf("Tap #%d\n", myTouch.tapCount);
    } else {
      myTouch.tapCount = 1;
      myTouch.firstTapTime = now;
      Serial.println("Tap #1 (new window)");
    }
  }
  if (!touchActive && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
  }
}
int myCheckTouchInput() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching) {
    if (now - myTouch.firstTapTime > myTouch.tapWindow) {
      int result = (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
      int count  = myTouch.tapCount;
      myResetTouchState();
      Serial.printf(result == 2 ? "LONG PRESS (%d taps)\n" : "TAP (%d tap%s)\n",
                    count, count > 1 ? "s" : "");
      return result;
    }
  }
  return 0;
}
void myCheckTouchBackground() { myUpdateTouchState(); }
int myPeekTouchAction() {
  myUpdateTouchState();
  unsigned long now = millis();
  if (myTouch.tapCount > 0 && !myTouch.isTouching)
    if (now - myTouch.firstTapTime > myTouch.tapWindow)
      return (myTouch.tapCount >= myTouch.longPressTaps) ? 2 : 1;
  return 0;
}

// ======================================================
// FORWARD DECLARATIONS
// ======================================================
void myAllocateMemory(bool resetMoments = true);
void mySaveWeights();
bool myLoadWeights(bool resetMoments = true);
void myExportHeader();
void myLoadConfig();
void mySaveConfig();
void myApplyDefaultConfig();
void myComputeArchSizes();
void mySyncLegacyVars();
void myActionCollect(int classIdx);
void myActionTrain();
void myActionInfer();
void myActionWebStream();
void myResetMenuState();
void myHandleMenuNavigation();
void myDrawMenu();
void myDispatchByIndex(int idx);
void myHandleStringCommand(const String& cmd);
void mySdListDir(const String& path);
void mySdReadText(const String& path);
void mySdReadJpeg(const String& path);
void mySdReadBinaryHead(const String& path, uint16_t numBytes);
bool mySdRemoveDirRecursive(const String& path, int& deleted);
void myCamCaptureSend();
void myBase64SendFrame(camera_fb_t* fb);
String myNormPath(String p);
void mySendHeatmap();
void myForwardPass(float* input);
void myMakeBoxTarget(float* map, const std::vector<FomoBox>& boxes);
void myMakeGaussianTarget(float* map, int cx, int cy);
void myBackwardFomoHead(int label, float* targetMap);
void myBackwardConv2();
void myBackwardPool1();
void myBackwardConv1();
void myUpdateWeights(int batchNum);
bool myLoadImageFromFile(const char* path, float* buf);
bool myLoadImageAugmented(const char* path, float* buf,
                           const std::vector<FomoBox>& srcBoxes,
                           std::vector<FomoBox>& outBoxes);
std::vector<FomoBox> myParseBoxesForFile(const String& jsonText, const String& filename);
String myLoadAnnotationsJson(const String& classLabel);
bool myEnsureParentDir(const String& path);

// ======================================================
// SETUP
// ======================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000);
  delay(1000);

  Serial.println("\n=== XIAO ESP32-S3 FOMO-WebSerial System Starting (v016) ===");
  Serial.printf("Free heap: %d bytes\n",  ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());

  pinMode(A0, INPUT);
  u8g2.begin();

  // Suppress ESP-IDF log spam (FB_OVF from camera) — must come before camera init
  esp_log_level_set("*",          ESP_LOG_WARN);
  esp_log_level_set("esp_camera", ESP_LOG_ERROR);

  // SD init — graceful degradation if absent
  pinMode(21, OUTPUT);
  digitalWrite(21, HIGH);
  delay(100);
  Serial.println("Checking SD card...");
  SPI.begin();
  SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if (!mySDavailable) {
    SD.end();
    Serial.println("No SD card — continuing without it");
    u8g2.firstPage();
    do { u8g2.setFont(u8g2_font_6x10_tf); u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000);
  } else {
    Serial.println("SD card mounted");
  }

  // Load config (compiled defaults if SD absent)
  if (mySDavailable) { myLoadConfig(); mySyncLegacyVars(); }
  else               { myApplyDefaultConfig(); mySyncLegacyVars(); }

  // RGB decode buffer (always 240×240 from camera)
  myRgbBuffer = (uint8_t*)ps_malloc(240 * 240 * 3);
  if (!myRgbBuffer) Serial.println("WARNING: RGB buffer alloc failed!");

  // Camera init
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_240X240;
  config.jpeg_quality = 12;
  config.fb_count     = 1;
  esp_camera_init(&config);
  Serial.println("Camera initialized");

  sensor_t* mySensor = esp_camera_sensor_get();
  if (mySensor != NULL) {
    mySensor->set_vflip(mySensor, 1);    // flip vertically
    mySensor->set_hmirror(mySensor, 1);  // flip horizontally
  }

  myAllocateMemory(true);

#ifdef USE_BAKED_WEIGHTS
  memcpy(myConv1_w,  myModel_conv1_w,  myCfg.conv1Weights * sizeof(float));
  memcpy(myConv1_b,  myModel_conv1_b,  myCfg.conv1Filters * sizeof(float));
  memcpy(myConv2_w,  myModel_conv2_w,  myCfg.conv2Weights * sizeof(float));
  memcpy(myConv2_b,  myModel_conv2_b,  myCfg.conv2Filters * sizeof(float));
  memcpy(myOutput_w, myModel_output_w, myCfg.outputWeights * sizeof(float));
  memcpy(myOutput_b, myModel_output_b, myCfg.numClasses    * sizeof(float));
  Serial.println("[weights] Baked-in weights loaded from myWeights.h");
  myWeightsTrained = true;
#endif

  if (mySDavailable && myLoadWeights(true)) {
    Serial.println("[weights] SD weights loaded — overriding baked-in weights");
  }

  myLastActivityTime = millis();
  myResetMenuState();
  delay(2000);
  Serial.println("System ready — Tap A0 to navigate, 3+ taps to select");
  myDrawMenu();
}

void loop() {
  myHandleMenuNavigation();
  // Continuous camera stream when CAM_STREAM active and menu is idle
  if (myCamStreaming && !myIsSelected &&
      (millis() - myLastCamStreamMs >= MY_CAM_STREAM_INTERVAL_MS)) {
    myLastCamStreamMs = millis();
    myCamCaptureSend();
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  CORE ML FUNCTIONS                                                        ██
// ██  myAllocateMemory, mySaveWeights, myLoadWeights                           ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// Helper macro: free-then-malloc in PSRAM
#define MY_REALLOC(ptr, size) \
  do { if (ptr) { free(ptr); ptr = nullptr; } \
       ptr = (float*)ps_malloc(size); } while(0)

#define MY_REALLOC_ZERO(ptr, size) \
  do { if (ptr) { free(ptr); ptr = nullptr; } \
       ptr = (float*)ps_malloc(size); \
       if (ptr) memset(ptr, 0, size); } while(0)

void myAllocateMemory(bool resetMoments) {
  MY_REALLOC(myConv1_w,  CONV1_WEIGHTS * sizeof(float));
  MY_REALLOC(myConv1_b,  CONV1_FILTERS * sizeof(float));
  MY_REALLOC(myConv2_w,  CONV2_WEIGHTS * sizeof(float));
  MY_REALLOC(myConv2_b,  CONV2_FILTERS * sizeof(float));
  MY_REALLOC(myOutput_w, myCfg.numClasses * myCfg.conv2Filters * sizeof(float));  // 1x1 conv only
  MY_REALLOC(myOutput_b, myCfg.numClasses * sizeof(float));

  MY_REALLOC(myConv1_w_grad,  CONV1_WEIGHTS * sizeof(float));
  MY_REALLOC(myConv1_b_grad,  CONV1_FILTERS * sizeof(float));
  MY_REALLOC(myConv2_w_grad,  CONV2_WEIGHTS * sizeof(float));
  MY_REALLOC(myConv2_b_grad,  CONV2_FILTERS * sizeof(float));
  MY_REALLOC(myOutput_w_grad, myCfg.numClasses * myCfg.conv2Filters * sizeof(float));
  MY_REALLOC(myOutput_b_grad, myCfg.numClasses * sizeof(float));

  MY_REALLOC(myConv1_output, CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  MY_REALLOC(myPool1_output, POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  MY_REALLOC(myConv2_output, CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS * sizeof(float));
  MY_REALLOC(myFomoMap,      FOMO_CELLS * myCfg.numClasses * sizeof(float));
  MY_REALLOC(myDense_output, myCfg.numClasses * sizeof(float));

  // Gradient propagation buffers
  MY_REALLOC(myFomoGrad,  FOMO_CELLS * CONV2_FILTERS * sizeof(float));
  MY_REALLOC(myConv2_grad, CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * CONV2_FILTERS * sizeof(float));
  MY_REALLOC(myPool1_grad, POOL1_OUTPUT_SIZE * POOL1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));
  MY_REALLOC(myConv1_grad, CONV1_OUTPUT_SIZE * CONV1_OUTPUT_SIZE * CONV1_FILTERS * sizeof(float));

  MY_REALLOC(myInputBuffer, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));

  if (resetMoments) {
    MY_REALLOC_ZERO(myConv1_w_m,  CONV1_WEIGHTS * sizeof(float));
    MY_REALLOC_ZERO(myConv1_w_v,  CONV1_WEIGHTS * sizeof(float));
    MY_REALLOC_ZERO(myConv1_b_m,  CONV1_FILTERS * sizeof(float));
    MY_REALLOC_ZERO(myConv1_b_v,  CONV1_FILTERS * sizeof(float));
    MY_REALLOC_ZERO(myConv2_w_m,  CONV2_WEIGHTS * sizeof(float));
    MY_REALLOC_ZERO(myConv2_w_v,  CONV2_WEIGHTS * sizeof(float));
    MY_REALLOC_ZERO(myConv2_b_m,  CONV2_FILTERS * sizeof(float));
    MY_REALLOC_ZERO(myConv2_b_v,  CONV2_FILTERS * sizeof(float));
    MY_REALLOC_ZERO(myOutput_w_m, myCfg.numClasses * myCfg.conv2Filters * sizeof(float));
    MY_REALLOC_ZERO(myOutput_w_v, myCfg.numClasses * myCfg.conv2Filters * sizeof(float));
    MY_REALLOC_ZERO(myOutput_b_m, myCfg.numClasses * sizeof(float));
    MY_REALLOC_ZERO(myOutput_b_v, myCfg.numClasses * sizeof(float));
    Serial.println("[mem] Adam moments reset (cold start)");
  } else {
    Serial.println("[mem] Adam moments preserved (warm resume)");
  }

  // Zero gradient and activation buffers
  if (myConv1_w_grad)  memset(myConv1_w_grad,  0, CONV1_WEIGHTS * sizeof(float));
  if (myConv1_b_grad)  memset(myConv1_b_grad,  0, CONV1_FILTERS * sizeof(float));
  if (myConv2_w_grad)  memset(myConv2_w_grad,  0, CONV2_WEIGHTS * sizeof(float));
  if (myConv2_b_grad)  memset(myConv2_b_grad,  0, CONV2_FILTERS * sizeof(float));
  if (myOutput_w_grad) memset(myOutput_w_grad,  0, myCfg.numClasses * myCfg.conv2Filters * sizeof(float));
  if (myOutput_b_grad) memset(myOutput_b_grad,  0, myCfg.numClasses * sizeof(float));

  // He initialisation
  if (myConv1_w) {
    float c1std = sqrt(2.0 / (9.0 * 3));
    for (int i = 0; i < CONV1_WEIGHTS; i++) myConv1_w[i] = ((float)rand()/RAND_MAX - 0.5f)*2.0f*c1std;
  }
  if (myConv1_b) memset(myConv1_b, 0, CONV1_FILTERS * sizeof(float));

  if (myConv2_w) {
    float c2std = sqrt(2.0 / (9.0 * CONV1_FILTERS));
    for (int i = 0; i < CONV2_WEIGHTS; i++) myConv2_w[i] = ((float)rand()/RAND_MAX - 0.5f)*2.0f*c2std;
  }
  if (myConv2_b) memset(myConv2_b, 0, CONV2_FILTERS * sizeof(float));

  if (myOutput_w) {
    float dstd = sqrt(2.0 / CONV2_FILTERS);
    int owSize = myCfg.numClasses * myCfg.conv2Filters;
    for (int i = 0; i < owSize; i++) myOutput_w[i] = ((float)rand()/RAND_MAX - 0.5f)*2.0f*dstd;
  }
  if (myOutput_b) memset(myOutput_b, 0, myCfg.numClasses * sizeof(float));

  if (!myInputBuffer || !myConv1_w || !myConv2_w || !myOutput_w ||
      !myConv1_output || !myPool1_output || !myConv2_output ||
      !myFomoMap || !myDense_output || !myFomoGrad) {
    Serial.println("FATAL: PSRAM allocation failed!");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "PSRAM ERROR!"); } while (u8g2.nextPage());
    while (1) { delay(1000); }
  }
  Serial.printf("[mem] Alloc done. Free PSRAM: %d bytes\n", ESP.getFreePsram());
  Serial.println("He-init random weights set");
}

#undef MY_REALLOC
#undef MY_REALLOC_ZERO

// ======================================================
// Helper: ensure the parent directory of a path exists
// ======================================================
bool myEnsureParentDir(const String& path) {
  int lastSlash = path.lastIndexOf('/');
  if (lastSlash <= 0) return true;
  String parent = path.substring(0, lastSlash);
  if (SD.exists(parent)) return true;
  // Create recursively (up to 3 levels)
  myEnsureParentDir(parent);
  return SD.mkdir(parent);
}

// ======================================================
// WEIGHT SAVE / LOAD / EXPORT
// Format: ASCII JSON header block + raw float32 binary
// ======================================================
void mySaveWeights() {
  if (!mySDavailable) { Serial.println("[weights] No SD card — cannot save"); return; }
  String filePath = "/header/" + myCfg.weightsFile;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File f = SD.open(filePath.c_str(), FILE_WRITE);
  if (!f) { Serial.println("[weights] ERROR: could not open file for writing"); return; }
  f.println("--- WEIGHTS HEADER BEGIN ---");
  JsonDocument doc;
  doc["version"]     = CURRENT_VERSION;
  doc["inputSize"]   = myCfg.inputSize;
  doc["numClasses"]  = myCfg.numClasses;
  doc["conv1Filters"]= myCfg.conv1Filters;
  doc["conv2Filters"]= myCfg.conv2Filters;
  doc["quantization"]= "float32";
  doc["modelType"]   = "fomo";
  JsonArray labels = doc["labels"].to<JsonArray>();
  for (int i = 0; i < myCfg.numLabels; i++) labels.add(myCfg.classLabels[i]);
  serializeJsonPretty(doc, f);
  f.println();
  f.println("--- WEIGHTS HEADER END ---");
  f.println();  // blank line before binary payload
  int owSize = myCfg.numClasses * myCfg.conv2Filters;
  f.write((uint8_t*)myConv1_w,  CONV1_WEIGHTS * sizeof(float));
  f.write((uint8_t*)myConv1_b,  CONV1_FILTERS * sizeof(float));
  f.write((uint8_t*)myConv2_w,  CONV2_WEIGHTS * sizeof(float));
  f.write((uint8_t*)myConv2_b,  CONV2_FILTERS * sizeof(float));
  f.write((uint8_t*)myOutput_w, owSize         * sizeof(float));
  f.write((uint8_t*)myOutput_b, myCfg.numClasses * sizeof(float));
  f.close();
  Serial.printf("[weights] Saved to %s\n", filePath.c_str());
  myExportHeader();
}

bool myLoadWeights(bool resetMoments) {
  if (!mySDavailable) { Serial.println("[weights] No SD card"); return false; }
  String filePath = "/header/" + myCfg.weightsFile;
  if (!SD.exists(filePath)) { Serial.println("[weights] No .bin file found"); return false; }
  Serial.printf("[weights] Loading from %s ...\n", filePath.c_str());
  File f = SD.open(filePath.c_str(), FILE_READ);
  if (!f) return false;
  // Skip the ASCII header block (read until blank line after "--- WEIGHTS HEADER END ---")
  int headerEnds = 0;
  while (f.available() && headerEnds < 2) {
    String line = f.readStringUntil('\n');
    line.trim();
    if (line == "--- WEIGHTS HEADER END ---") headerEnds = 1;
    else if (headerEnds == 1 && line.length() == 0) headerEnds = 2;
  }
  int owSize = myCfg.numClasses * myCfg.conv2Filters;
  f.read((uint8_t*)myConv1_w,  CONV1_WEIGHTS    * sizeof(float));
  f.read((uint8_t*)myConv1_b,  CONV1_FILTERS    * sizeof(float));
  f.read((uint8_t*)myConv2_w,  CONV2_WEIGHTS    * sizeof(float));
  f.read((uint8_t*)myConv2_b,  CONV2_FILTERS    * sizeof(float));
  f.read((uint8_t*)myOutput_w, owSize            * sizeof(float));
  f.read((uint8_t*)myOutput_b, myCfg.numClasses  * sizeof(float));
  f.close();
  Serial.println("[weights] Loaded successfully");
  myWeightsTrained = true;
  return true;
}

void myExportHeader() {
  if (!mySDavailable) return;
  if (!SD.exists("/header")) SD.mkdir("/header");
  File file = SD.open("/header/myWeights.h", FILE_WRITE);
  if (!file) return;
  file.println("#ifndef MY_MODEL_H\n#define MY_MODEL_H");
  file.println("// FOMO model weights — copy to sketch folder then uncomment #define USE_BAKED_WEIGHTS");
  file.printf("// #define NUM_CLASSES %d\n", myCfg.numClasses);
  auto myDump = [&](const char* name, float* data, int size) {
    file.printf("const float %s[] = { ", name);
    for (int i = 0; i < size; i++) {
      file.print(data[i], 6); file.print("f");
      if (i < size-1) file.print(", ");
      if ((i+1)%8 == 0) file.println();
    }
    file.println(" };");
  };
  int owSize = myCfg.numClasses * myCfg.conv2Filters;
  myDump("myModel_conv1_w",  myConv1_w,  CONV1_WEIGHTS);
  myDump("myModel_conv1_b",  myConv1_b,  CONV1_FILTERS);
  myDump("myModel_conv2_w",  myConv2_w,  CONV2_WEIGHTS);
  myDump("myModel_conv2_b",  myConv2_b,  CONV2_FILTERS);
  myDump("myModel_output_w", myOutput_w, owSize);
  myDump("myModel_output_b", myOutput_b, myCfg.numClasses);
  file.println("#endif");
  file.close();
  Serial.println("[weights] Exported /header/myWeights.h");
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 1: IMAGE COLLECTION                                                 ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myRenderRgbToOLED(int imageCount) {
  int myOledWidth  = u8g2.getDisplayWidth();
  int myOledHeight = u8g2.getDisplayHeight();
  int myScaleX = 240 / myOledWidth;
  int myScaleY = 240 / myOledHeight;
  u8g2.firstPage();
  do {
    for (int myOledX = 0; myOledX < myOledWidth; myOledX++) {
      for (int myOledY = 0; myOledY < myOledHeight; myOledY++) {
        size_t myPixelIndex = ((myOledY * myScaleY) * 240 + (myOledX * myScaleX)) * 3;
        uint8_t myBrightness = (myRgbBuffer[myPixelIndex]   +
                                myRgbBuffer[myPixelIndex+1] +
                                myRgbBuffer[myPixelIndex+2]) / 3;
        if (myBrightness > 100) u8g2.drawPixel(myOledX, myOledY);
      }
    }
    if (imageCount >= 0) {
      u8g2.setFont(u8g2_font_ncenB10_tr);
      u8g2.setColorIndex(0); u8g2.drawBox(0, 0, 20, 15);
      u8g2.setColorIndex(1); u8g2.setCursor(3, 10); u8g2.print(String(imageCount));
    } else {
      u8g2.setFont(u8g2_font_5x7_tf);
      u8g2.setColorIndex(0); u8g2.drawBox(50, 0, 22, 8);
      u8g2.setColorIndex(1); u8g2.drawStr(52, 7, "LIVE");
    }
  } while (u8g2.nextPage());
}

void myDisplayImageOnOLED(camera_fb_t* fb, int imageCount) {
  if (!myRgbBuffer) return;
  if (!fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) return;
  myFlipImageHorizontal();
  myRenderRgbToOLED(imageCount);
}

void myActionCollect(int classIdx) {
  if (!mySDavailable) {
    Serial.println("No SD card — cannot collect images");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.printf("\n>>> Collection mode: %s\n", myClassLabels[classIdx].c_str());
  Serial.println("  TAP (1-2) = Capture   LONG PRESS (3+) = Exit");
  Serial.println("  Serial: 'T'=capture   'L'=exit");

  myResetTouchState();

  String path = "/images/" + myClassLabels[classIdx];
  if (!SD.exists("/images")) SD.mkdir("/images");
  if (!SD.exists(path)) SD.mkdir(path);

  int counts[8] = {};
  File root = SD.open("/images/" + myClassLabels[classIdx]);
  if (root) {
    while (File file = root.openNextFile()) {
      if (!file.isDirectory()) {
        String fn = String(file.name());
        if (fn.endsWith(".jpg") || fn.endsWith(".JPG")) counts[classIdx]++;
      }
      file.close();
    }
    root.close();
  }

  unsigned long lastCameraDrain = 0;
  unsigned long lastOLED        = 0;
  bool oledNeedsUpdate          = false;
  bool shouldCapture            = false;

  while (true) {
    unsigned long now = millis();

    // Keep SD browser alive during collection
    if (Serial.available()) {
      String line = "";
      char c = Serial.read();
      if (c == 'l' || c == 'L') { myResetMenuState(); return; }
      else if (c == 't' || c == 'T') { shouldCapture = true; }
      else {
        line += c;
        unsigned long lineStart = millis();
        while (millis() - lineStart < 20 && Serial.available()) {
          char nc = Serial.read();
          if (nc == '\n') { line.trim(); if (line.length() > 0) myHandleStringCommand(line); break; }
          line += nc;
        }
      }
    }

    if (now - lastCameraDrain > 50) {
      lastCameraDrain = now;
      if (!shouldCapture) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
          if (now - lastOLED > 250 && myRgbBuffer) {
            if (fmt2rgb888(fb->buf, fb->len, fb->format, myRgbBuffer)) {
              myFlipImageHorizontal();
              oledNeedsUpdate = true; lastOLED = now;
            }
          }
          esp_camera_fb_return(fb);
        }
      }
    }

    if (oledNeedsUpdate) { oledNeedsUpdate = false; myRenderRgbToOLED(-1); }

    int touchAction = myCheckTouchInput();
    if (touchAction == 2) { Serial.println("Exiting collection"); myResetMenuState(); return; }
    else if (touchAction == 1) { shouldCapture = true; }

    if (shouldCapture) {
      shouldCapture = false;
      camera_fb_t* fb = esp_camera_fb_get();
      if (fb) {
        String fileName = path + "/img_" + String(millis()) + ".jpg";
        File file = SD.open(fileName, FILE_WRITE);
        if (file) {
          file.write(fb->buf, fb->len);
          file.close();
          counts[classIdx]++;
          Serial.printf("Saved: %s (Total: %d)\n", fileName.c_str(), counts[classIdx]);
          // WebSerial webpage sample counter update
          Serial.printf("COLLECT_COUNT:%d,%d\n", classIdx, counts[classIdx]);
          myDisplayImageOnOLED(fb, counts[classIdx]);
          // Also send frame to browser for preview
          myBase64SendFrame(fb);
          delay(300);
          lastOLED = millis();
        }
        esp_camera_fb_return(fb);
      }
    }
    delay(5);
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2: FORWARD PASS, FOMO HEAD, BACKWARD PASS, OPTIMIZER               ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// FORWARD PASS
// Fills: myConv1_output, myPool1_output, myConv2_output,
//        myFomoMap[numClasses][fomoCells] (sigmoid),
//        myDense_output[numClasses]  (global average pool of myFomoMap)
// ======================================================
void myForwardPass(float* input) {
  int iSz = myCfg.inputSize;
  int c1o  = myCfg.conv1OutputSize;
  int p1o  = myCfg.pool1OutputSize;
  int c2o  = myCfg.conv2OutputSize;
  int c1f  = myCfg.conv1Filters;
  int c2f  = myCfg.conv2Filters;
  int cells = myCfg.fomoCells;

  // Conv1: iSz x iSz x 3  ->  c1o x c1o x c1f
  for (int f = 0; f < c1f; f++) {
    int ob = f * c1o * c1o;
    for (int y = 0; y < c1o; y++) {
      for (int x = 0; x < c1o; x++) {
        float sum = myConv1_b[f];
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            int inPos = ((y+ky)*iSz + (x+kx)) * 3;
            int wPos  = f*27 + ky*9 + kx*3;
            sum += input[inPos]   * myConv1_w[wPos]   +
                   input[inPos+1] * myConv1_w[wPos+1] +
                   input[inPos+2] * myConv1_w[wPos+2];
          }
        }
        myConv1_output[ob + y*c1o + x] = leaky_relu(clip_value(sum));
      }
    }
  }

  // Pool1: 2x2 max-pool  ->  p1o x p1o x c1f
  for (int f = 0; f < c1f; f++) {
    int ib = f * c1o * c1o;
    int ob = f * p1o * p1o;
    for (int y = 0; y < p1o; y++) {
      for (int x = 0; x < p1o; x++) {
        int iy = y*2, ix = x*2;
        float mv = myConv1_output[ib + iy*c1o + ix];
        mv = max(mv, myConv1_output[ib + iy*c1o     + ix+1]);
        mv = max(mv, myConv1_output[ib + (iy+1)*c1o + ix]);
        mv = max(mv, myConv1_output[ib + (iy+1)*c1o + ix+1]);
        myPool1_output[ob + y*p1o + x] = mv;
      }
    }
  }

  // Conv2: p1o x p1o x c1f  ->  c2o x c2o x c2f
  for (int f = 0; f < c2f; f++) {
    int ob = f * c2o * c2o;
    for (int y = 0; y < c2o; y++) {
      for (int x = 0; x < c2o; x++) {
        float sum = myConv2_b[f];
        for (int c = 0; c < c1f; c++) {
          int ib = c * p1o * p1o;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              sum += myPool1_output[ib + (y+ky)*p1o + (x+kx)] *
                     myConv2_w[f * c1f*9 + c*9 + ky*3 + kx];
            }
          }
        }
        myConv2_output[ob + y*c2o + x] = leaky_relu(clip_value(sum));
      }
    }
  }

  // FOMO head: 1x1 conv over conv2 spatial map -> myFomoMap[cls][cell]
  // Weight layout: myOutput_w[cls * conv2Filters + f]
  for (int cls = 0; cls < myCfg.numClasses; cls++) {
    float* map = myFomoMap + cls * cells;
    for (int cell = 0; cell < cells; cell++) {
      float sum = myOutput_b[cls];
      for (int f = 0; f < c2f; f++) {
        sum += myConv2_output[f * cells + cell] * myOutput_w[cls * c2f + f];
      }
      map[cell] = my_sigmoid(clip_value(sum, -15, 15));
    }
  }

  // Global average pool of FOMO map -> myDense_output (used for class prediction)
  for (int cls = 0; cls < myCfg.numClasses; cls++) {
    float* map = myFomoMap + cls * cells;
    float s = 0;
    for (int cell = 0; cell < cells; cell++) s += map[cell];
    myDense_output[cls] = s / cells;
  }
}

// ======================================================
// BOX TARGET MAP
// ======================================================
void myMakeBoxTarget(float* map, const std::vector<FomoBox>& boxes) {
  int grid  = myCfg.fomoGrid;
  int cells = myCfg.fomoCells;
  memset(map, 0, cells * sizeof(float));
  for (const FomoBox& b : boxes) {
    int gx1 = constrain((int)(b.x1 * grid), 0, grid-1);
    int gy1 = constrain((int)(b.y1 * grid), 0, grid-1);
    int gx2 = constrain((int)(b.x2 * grid), 0, grid-1);
    int gy2 = constrain((int)(b.y2 * grid), 0, grid-1);
    for (int gy = gy1; gy <= gy2; gy++)
      for (int gx = gx1; gx <= gx2; gx++)
        map[gy * grid + gx] = 1.0f;
  }
}

void myMakeGaussianTarget(float* map, int cx, int cy) {
  int grid = myCfg.fomoGrid;
  FomoBox b;
  b.x1 = (float)cx       / grid;
  b.y1 = (float)cy       / grid;
  b.x2 = (float)(cx + 1) / grid;
  b.y2 = (float)(cy + 1) / grid;
  std::vector<FomoBox> v = { b };
  myMakeBoxTarget(map, v);
}

// ======================================================
// BACKWARD — FOMO HEAD
// ======================================================
void myBackwardFomoHead(int label, float* targetMap) {
  int c2f   = myCfg.conv2Filters;
  int cells = myCfg.fomoCells;
  memset(myConv2_grad, 0, CONV2_OUTPUT_SIZE * CONV2_OUTPUT_SIZE * c2f * sizeof(float));
  for (int cls = 0; cls < myCfg.numClasses; cls++) {
    float* map = myFomoMap + cls * cells;
    for (int cell = 0; cell < cells; cell++) {
      float tgt  = (cls == label) ? targetMap[cell] : 0.0f;
      float diff = (map[cell] - tgt) * my_sigmoid_deriv(map[cell]);
      float dL   = 2.0f * diff / (float)(cells * myCfg.numClasses);
      myOutput_b_grad[cls] += dL;
      for (int f = 0; f < c2f; f++) {
        myOutput_w_grad[cls * c2f + f] += dL * myConv2_output[f * cells + cell];
        myConv2_grad[f * cells + cell] += dL * myOutput_w[cls * c2f + f];
      }
    }
  }
}

// ======================================================
// BACKWARD — CONV2
// ======================================================
void myBackwardConv2() {
  int c2o = myCfg.conv2OutputSize;
  int p1o = myCfg.pool1OutputSize;
  int c1f = myCfg.conv1Filters;
  int c2f = myCfg.conv2Filters;
  for (int i = 0; i < c2o * c2o * c2f; i++)
    myConv2_grad[i] *= leaky_relu_deriv(myConv2_output[i]);
  memset(myPool1_grad, 0, p1o * p1o * c1f * sizeof(float));
  for (int f = 0; f < c2f; f++) {
    int ob = f * c2o * c2o;
    for (int y = 0; y < c2o; y++) {
      for (int x = 0; x < c2o; x++) {
        float grad = myConv2_grad[ob + y*c2o + x];
        myConv2_b_grad[f] += grad;
        for (int c = 0; c < c1f; c++) {
          int ib = c * p1o * p1o;
          for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
              int pi = ib + (y+ky)*p1o + (x+kx);
              int wi = f * c1f*9 + c*9 + ky*3 + kx;
              myConv2_w_grad[wi] += grad * myPool1_output[pi];
              myPool1_grad[pi]   += grad * myConv2_w[wi];
            }
          }
        }
      }
    }
  }
}

// ======================================================
// BACKWARD — POOL1
// ======================================================
void myBackwardPool1() {
  int c1o = myCfg.conv1OutputSize;
  int p1o = myCfg.pool1OutputSize;
  int c1f = myCfg.conv1Filters;
  memset(myConv1_grad, 0, c1o * c1o * c1f * sizeof(float));
  for (int f = 0; f < c1f; f++) {
    int ib = f * c1o * c1o;
    int ob = f * p1o * p1o;
    for (int y = 0; y < p1o; y++) {
      for (int x = 0; x < p1o; x++) {
        int iy = y*2, ix = x*2;
        float poolVal = myPool1_output[ob + y*p1o + x];
        float grad    = myPool1_grad[ob + y*p1o + x];
        if (myConv1_output[ib + iy*c1o     + ix  ] == poolVal) myConv1_grad[ib + iy*c1o     + ix  ] += grad;
        if (myConv1_output[ib + iy*c1o     + ix+1] == poolVal) myConv1_grad[ib + iy*c1o     + ix+1] += grad;
        if (myConv1_output[ib + (iy+1)*c1o + ix  ] == poolVal) myConv1_grad[ib + (iy+1)*c1o + ix  ] += grad;
        if (myConv1_output[ib + (iy+1)*c1o + ix+1] == poolVal) myConv1_grad[ib + (iy+1)*c1o + ix+1] += grad;
      }
    }
  }
}

// ======================================================
// BACKWARD — CONV1
// ======================================================
void myBackwardConv1() {
  int iSz = myCfg.inputSize;
  int c1o = myCfg.conv1OutputSize;
  int c1f = myCfg.conv1Filters;
  for (int i = 0; i < c1o * c1o * c1f; i++)
    myConv1_grad[i] *= leaky_relu_deriv(myConv1_output[i]);
  for (int f = 0; f < c1f; f++) {
    int ob = f * c1o * c1o;
    for (int y = 0; y < c1o; y++) {
      for (int x = 0; x < c1o; x++) {
        float grad = myConv1_grad[ob + y*c1o + x];
        myConv1_b_grad[f] += grad;
        for (int ky = 0; ky < 3; ky++) {
          for (int kx = 0; kx < 3; kx++) {
            int inPos = ((y+ky)*iSz + (x+kx)) * 3;
            int wPos  = f*27 + ky*9 + kx*3;
            myConv1_w_grad[wPos]   += grad * myInputBuffer[inPos];
            myConv1_w_grad[wPos+1] += grad * myInputBuffer[inPos+1];
            myConv1_w_grad[wPos+2] += grad * myInputBuffer[inPos+2];
          }
        }
      }
    }
  }
}

// ======================================================
// ADAM OPTIMIZER
// ======================================================
void myAdamUpdate(float* w, float* g, float* m, float* v, int size, int step) {
  float b1 = 0.9f, b2 = 0.999f, eps = 1e-6f;
  float lr_t = LEARNING_RATE * sqrtf(1 - powf(b2, step)) / (1 - powf(b1, step));
  for (int i = 0; i < size; i++) {
    m[i] = b1*m[i] + (1-b1)*g[i];
    v[i] = b2*v[i] + (1-b2)*g[i]*g[i];
    w[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    w[i]  = clip_value(w[i], -10, 10);
  }
}

void myUpdateWeights(int step) {
  int owSize = myCfg.numClasses * myCfg.conv2Filters;
  myAdamUpdate(myConv1_w,  myConv1_w_grad,  myConv1_w_m,  myConv1_w_v,  CONV1_WEIGHTS, step);
  myAdamUpdate(myConv1_b,  myConv1_b_grad,  myConv1_b_m,  myConv1_b_v,  CONV1_FILTERS, step);
  myAdamUpdate(myConv2_w,  myConv2_w_grad,  myConv2_w_m,  myConv2_w_v,  CONV2_WEIGHTS, step);
  myAdamUpdate(myConv2_b,  myConv2_b_grad,  myConv2_b_m,  myConv2_b_v,  CONV2_FILTERS, step);
  myAdamUpdate(myOutput_w, myOutput_w_grad, myOutput_w_m, myOutput_w_v, owSize,        step);
  myAdamUpdate(myOutput_b, myOutput_b_grad, myOutput_b_m, myOutput_b_v, myCfg.numClasses, step);
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2b: IMAGE LOADING (SD + AUGMENTATION) + ANNOTATION PARSER          ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

bool myLoadImageFromFile(const char* path, float* buf) {
  File f = SD.open(path);
  if (!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if (!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  if (!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if (!ok) return false;
  myFlipImageHorizontal();
  int iSz = myCfg.inputSize;
  for (int y = 0; y < iSz; y++) {
    for (int x = 0; x < iSz; x++) {
      int sy = (int)((y+0.5)*240.0/iSz); if (sy > 239) sy = 239;
      int sx = (int)((x+0.5)*240.0/iSz); if (sx > 239) sx = 239;
      int srcIdx = (sy*240 + sx)*3;
      int dstIdx = (y*iSz + x)*3;
      buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
      buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
      buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
    }
  }
  return true;
}

bool myLoadImageAugmented(const char* path, float* buf,
                           const std::vector<FomoBox>& srcBoxes,
                           std::vector<FomoBox>& outBoxes) {
  File f = SD.open(path);
  if (!f) return false;
  size_t sz = f.size();
  uint8_t* jpg = (uint8_t*)ps_malloc(sz);
  if (!jpg) { f.close(); return false; }
  f.read(jpg, sz);
  f.close();
  if (!myRgbBuffer) { free(jpg); return false; }
  bool ok = fmt2rgb888(jpg, sz, PIXFORMAT_JPEG, myRgbBuffer);
  free(jpg);
  if (!ok) return false;
  myFlipImageHorizontal();

  int iSz = myCfg.inputSize;
  if (myCfg.useAugmentation) {
    const int MY_AUG_CROP_MIN = 160;
    const int MY_AUG_CROP_MAX = 220;
    int cropSize  = MY_AUG_CROP_MIN + random(MY_AUG_CROP_MAX - MY_AUG_CROP_MIN + 1);
    int maxOffset = 240 - cropSize;
    int cropX = (maxOffset > 0) ? random(maxOffset + 1) : 0;
    int cropY = (maxOffset > 0) ? random(maxOffset + 1) : 0;
    outBoxes.clear();
    for (const FomoBox& b : srcBoxes) {
      float px1 = b.x1 * 240.0f, py1 = b.y1 * 240.0f;
      float px2 = b.x2 * 240.0f, py2 = b.y2 * 240.0f;
      float cx1 = constrain(px1, (float)cropX, (float)(cropX + cropSize));
      float cy1 = constrain(py1, (float)cropY, (float)(cropY + cropSize));
      float cx2 = constrain(px2, (float)cropX, (float)(cropX + cropSize));
      float cy2 = constrain(py2, (float)cropY, (float)(cropY + cropSize));
      if (cx2 - cx1 < 1.0f || cy2 - cy1 < 1.0f) continue;
      FomoBox ob;
      ob.x1 = (cx1 - cropX) / cropSize;
      ob.y1 = (cy1 - cropY) / cropSize;
      ob.x2 = (cx2 - cropX) / cropSize;
      ob.y2 = (cy2 - cropY) / cropSize;
      outBoxes.push_back(ob);
    }
    for (int y = 0; y < iSz; y++) {
      for (int x = 0; x < iSz; x++) {
        int sy = cropY + (int)((y + 0.5f) * cropSize / iSz);
        int sx = cropX + (int)((x + 0.5f) * cropSize / iSz);
        sy = constrain(sy, 0, 239); sx = constrain(sx, 0, 239);
        int srcIdx = (sy * 240 + sx) * 3;
        int dstIdx = (y * iSz + x) * 3;
        buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
        buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
        buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
      }
    }
  } else {
    for (int y = 0; y < iSz; y++) {
      for (int x = 0; x < iSz; x++) {
        int sy = (int)((y + 0.5f) * 240.0f / iSz); if (sy > 239) sy = 239;
        int sx = (int)((x + 0.5f) * 240.0f / iSz); if (sx > 239) sx = 239;
        int srcIdx = (sy * 240 + sx) * 3;
        int dstIdx = (y * iSz + x) * 3;
        buf[dstIdx]   = myRgbBuffer[srcIdx]   / 255.0f;
        buf[dstIdx+1] = myRgbBuffer[srcIdx+1] / 255.0f;
        buf[dstIdx+2] = myRgbBuffer[srcIdx+2] / 255.0f;
      }
    }
    outBoxes = srcBoxes;
  }
  return true;
}

std::vector<FomoBox> myParseBoxesForFile(const String& jsonText, const String& filename) {
  std::vector<FomoBox> result;
  int keyPos = jsonText.indexOf("\"" + filename + "\"");
  if (keyPos < 0) return result;
  int arrOpen  = jsonText.indexOf('[', keyPos);
  if (arrOpen < 0) return result;
  int arrClose = jsonText.indexOf(']', arrOpen);
  if (arrClose < 0) return result;
  String arrText = jsonText.substring(arrOpen, arrClose + 1);
  int pos = 0;
  while (true) {
    int objOpen  = arrText.indexOf('{', pos);
    if (objOpen < 0) break;
    int objClose = arrText.indexOf('}', objOpen);
    if (objClose < 0) break;
    String obj = arrText.substring(objOpen, objClose + 1);
    auto myGetVal = [&](const char* key) -> float {
      int k = obj.indexOf(key);
      if (k < 0) return 0.0f;
      int colon = obj.indexOf(':', k);
      if (colon < 0) return 0.0f;
      return obj.substring(colon + 1).toFloat();
    };
    FomoBox b;
    b.x1 = myGetVal("x1"); b.y1 = myGetVal("y1");
    b.x2 = myGetVal("x2"); b.y2 = myGetVal("y2");
    if (b.x2 > b.x1 && b.y2 > b.y1) result.push_back(b);
    pos = objClose + 1;
  }
  return result;
}

String myLoadAnnotationsJson(const String& classLabel) {
  String jsonPath = "/images/" + classLabel + "/annotations.json";
  if (!SD.exists(jsonPath)) return "";
  File f = SD.open(jsonPath, FILE_READ);
  if (!f) return "";
  String text = "";
  while (f.available()) text += (char)f.read();
  f.close();
  return text;
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 2c: TRAINING FUNCTION                                               ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myActionTrain() {
  if (!mySDavailable) {
    Serial.println("No SD card — cannot train");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "No SD card"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Training mode (FOMO)");
  Serial.println("  During training: 1 tap = Save+exit  |  3+ taps = Exit without save");
  Serial.println("  Serial: 'L'=save+exit");

  myResetTouchState();

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 12, "TRAINING");
    u8g2.drawStr(0, 24, "Loading...");
  } while (u8g2.nextPage());

  if (myLoadWeights(false)) Serial.println("Continuing from saved weights (warm Adam)");
  else                      Serial.println("Starting fresh training");

  while (true) {
    myTrainingData.clear();
    for (int i = 0; i < myCfg.numClasses; i++) {
      String jsonText = (i == 0) ? "" : myLoadAnnotationsJson(myCfg.classLabels[i]);
      bool hasJson = (jsonText.length() > 0);

      File root = SD.open("/images/" + myCfg.classLabels[i]);
      if (root) {
        while (File file = root.openNextFile()) {
          if (!file.isDirectory()) {
            String fn = String(file.name());
            if (fn.endsWith(".jpg") || fn.endsWith(".JPG")) {
              TrainingItem item;
              item.path  = String(file.path());
              item.label = i;
              if (i == 0) {
                // background — all-zero target map
              } else if (hasJson) {
                item.boxes = myParseBoxesForFile(jsonText, fn);
                if (item.boxes.empty()) item.boxes.push_back(myDefaultBox);
              } else {
                item.boxes.push_back(myDefaultBox);
              }
              myTrainingData.push_back(item);
            }
          }
          file.close();
        }
        root.close();
      }
      if (hasJson) Serial.printf("Class %s: annotations.json loaded\n", myCfg.classLabels[i].c_str());
    }

    if (myTrainingData.empty()) {
      u8g2.firstPage();
      do { u8g2.drawStr(0, 20, "No Images!"); } while (u8g2.nextPage());
      delay(2000); myResetMenuState(); return;
    }

    std::sort(myTrainingData.begin(), myTrainingData.end(),
              [](const TrainingItem& a, const TrainingItem& b){ return a.path < b.path; });

    std::vector<TrainingItem> myValidationData;
    if (myCfg.validationImages > 0) {
      int counts[8] = {}, skip[8] = {};
      for (auto& item : myTrainingData) counts[item.label]++;
      for (int c = 0; c < myCfg.numClasses; c++) skip[c] = min(myCfg.validationImages, counts[c]);

      std::vector<TrainingItem> trainOnly;
      int seen[8] = {};
      for (int i = (int)myTrainingData.size() - 1; i >= 0; i--) {
        int c = myTrainingData[i].label;
        if (seen[c] < skip[c]) { myValidationData.push_back(myTrainingData[i]); seen[c]++; }
        else                    { trainOnly.push_back(myTrainingData[i]); }
      }
      myTrainingData = trainOnly;
      Serial.printf("Val: %d  Train: %d\n",
                    (int)myValidationData.size(), (int)myTrainingData.size());
    }

    int total           = myTrainingData.size();
    int batchesPerEpoch = (total + BATCH_SIZE - 1) / BATCH_SIZE;
    int totalBatches    = TARGET_EPOCHS * batchesPerEpoch;
    Serial.printf("Training: %d images, %d total batches\n", total, totalBatches);

    std::vector<int> indices;
    for (int i = 0; i < total; i++) indices.push_back(i);

    float runningLoss = 0; int lossCount = 0;
    int epochLossCount = 0; float epochLoss = 0; int epochCorrect = 0; int epochTotal = 0;

    float* myTargetMap = (float*)ps_malloc(FOMO_CELLS * sizeof(float));
    if (!myTargetMap) { Serial.println("OOM: targetMap"); myResetMenuState(); return; }

    for (int batch = 0; batch < totalBatches; batch++) {

      // Epoch start: shuffle + send TRAIN_EPOCH for previous epoch
      if (batch % batchesPerEpoch == 0) {
        int epoch = batch / batchesPerEpoch;
        if (epoch > 0 && epochTotal > 0) {
          float epochAvgLoss = epochLoss / epochLossCount;
          float epochAcc     = (float)epochCorrect / epochTotal;
          // WebSerial epoch token
          Serial.printf("TRAIN_EPOCH:%d,%.4f,%.4f\n", epoch, epochAvgLoss, epochAcc);
        }
        epochLoss = 0; epochLossCount = 0; epochCorrect = 0; epochTotal = 0;
        Serial.printf("\n--- Epoch %d/%d ---\n", epoch + 1, TARGET_EPOCHS);
        for (int i = total - 1; i > 0; i--) {
          int j = random(i + 1); int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }
      }

      // Serial command check (L = save+exit)
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'l' || c == 'L' || c == 'x' || c == 'X') {
          Serial.println("Stopping training (save+exit)...");
          free(myTargetMap);
          mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
        }
      }

      // Touch: single tap = save+exit, long press = exit without save
      myCheckTouchBackground();
      if (myPeekTouchAction() == 1) {
        myCheckTouchInput();
        Serial.println("Tap — saving weights and exiting training");
        free(myTargetMap);
        mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
      }
      if (myPeekTouchAction() == 2) {
        myCheckTouchInput();
        Serial.println("Long press — stopping training");
        free(myTargetMap);
        mySaveWeights(); myWeightsTrained = true; myResetMenuState(); return;
      }

      int batchStart = (batch % batchesPerEpoch) * BATCH_SIZE;
      int batchEnd   = min(batchStart + BATCH_SIZE, total);

      float batchLoss = 0; int correctCount = 0;

      // Zero ALL weight gradient buffers once per batch
      int owSize = myCfg.numClasses * myCfg.conv2Filters;
      memset(myConv1_w_grad,  0, CONV1_WEIGHTS * sizeof(float));
      memset(myConv1_b_grad,  0, CONV1_FILTERS * sizeof(float));
      memset(myConv2_w_grad,  0, CONV2_WEIGHTS * sizeof(float));
      memset(myConv2_b_grad,  0, CONV2_FILTERS * sizeof(float));
      memset(myOutput_w_grad, 0, owSize         * sizeof(float));
      memset(myOutput_b_grad, 0, myCfg.numClasses * sizeof(float));

      for (int i = batchStart; i < batchEnd; i++) {
        int idx = indices[i];
        TrainingItem& img = myTrainingData[idx];

        std::vector<FomoBox> croppedBoxes;
        if (!myLoadImageAugmented(img.path.c_str(), myInputBuffer, img.boxes, croppedBoxes)) continue;

        myForwardPass(myInputBuffer);

        if (img.label == 0 || croppedBoxes.empty()) {
          memset(myTargetMap, 0, FOMO_CELLS * sizeof(float));
        } else {
          myMakeBoxTarget(myTargetMap, croppedBoxes);
        }

        // MSE loss over FOMO map
        float loss = 0;
        for (int cls = 0; cls < myCfg.numClasses; cls++) {
          float* map = myFomoMap + cls * FOMO_CELLS;
          for (int cell = 0; cell < FOMO_CELLS; cell++) {
            float tgt = (cls == img.label) ? myTargetMap[cell] : 0.0f;
            float diff = map[cell] - tgt;
            loss += diff * diff;
          }
        }
        loss /= (FOMO_CELLS * myCfg.numClasses);
        batchLoss += loss;

        int pred = 0;
        for (int cls = 1; cls < myCfg.numClasses; cls++)
          if (myDense_output[cls] > myDense_output[pred]) pred = cls;
        if (pred == img.label) correctCount++;

        myBackwardFomoHead(img.label, myTargetMap);
        myBackwardConv2();
        myBackwardPool1();
        myBackwardConv1();
      }

      myUpdateWeights(batch + 1);

      float avgLoss  = batchLoss / (batchEnd - batchStart);
      float batchAcc = (float)correctCount / (batchEnd - batchStart);
      runningLoss  += avgLoss; lossCount++;
      epochLoss    += avgLoss; epochLossCount++;
      epochCorrect += correctCount; epochTotal += (batchEnd - batchStart);

      if ((batch+1) % 5 == 0) {
        float displayLoss = runningLoss / lossCount;
        u8g2.firstPage();
        do {
          u8g2.setFont(u8g2_font_5x7_tf);
          u8g2.setCursor(0, 12); u8g2.print("Training...");
          u8g2.setCursor(0, 24);
          u8g2.print("B:"); u8g2.print(batch+1); u8g2.print("/"); u8g2.print(totalBatches);
          u8g2.setCursor(0, 36);
          u8g2.print("L:"); u8g2.print(displayLoss, 3);
          u8g2.print(" A:"); u8g2.print((int)(batchAcc*100)); u8g2.print("%");
        } while (u8g2.nextPage());
        runningLoss = 0; lossCount = 0;
      }
      if ((batch+1) % 10 == 0) {
        Serial.printf("Batch %d/%d  Loss:%.4f  Acc:%.1f%%\n",
                      batch+1, totalBatches, avgLoss, batchAcc*100);
      }
    }

    free(myTargetMap);
    Serial.println("\n--- Training Complete ---");

    // Final epoch serial token
    if (epochTotal > 0) {
      float epochAvgLoss = epochLoss / epochLossCount;
      float epochAcc     = (float)epochCorrect / epochTotal;
      Serial.printf("TRAIN_EPOCH:%d,%.4f,%.4f\n", TARGET_EPOCHS, epochAvgLoss, epochAcc);
    }

    // Validation pass
    if (!myValidationData.empty()) {
      int valCorrect = 0, valCount = 0;
      for (auto& vitem : myValidationData) {
        if (!myLoadImageFromFile(vitem.path.c_str(), myInputBuffer)) continue;
        myForwardPass(myInputBuffer);
        int pred = 0;
        for (int j = 1; j < myCfg.numClasses; j++)
          if (myDense_output[j] > myDense_output[pred]) pred = j;
        if (pred == vitem.label) valCorrect++;
        valCount++;
      }
      if (valCount > 0)
        Serial.printf("Validation Accuracy: %.1f%%  (%d/%d)\n",
                      100.0f * valCorrect / valCount, valCorrect, valCount);
    }

    mySaveWeights();
    myWeightsTrained = true;

    u8g2.firstPage();
    do {
      u8g2.drawStr(0, 12, "DONE!");
      u8g2.drawStr(0, 24, "Tap:Again");
      u8g2.drawStr(0, 36, "3+Taps:Exit");
    } while (u8g2.nextPage());

    myResetTouchState();
    Serial.println("Training done. T=train again  L=exit");
    while (true) {
      if (Serial.available()) {
        char c = Serial.read();
        if (c == 'l' || c == 'L') { myResetMenuState(); return; }
        else if (c == 't' || c == 'T') { break; }
      }
      int touchAction = myCheckTouchInput();
      if (touchAction == 2) { myResetMenuState(); return; }
      else if (touchAction == 1) { Serial.println("Starting new training cycle"); break; }
      delay(10);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 3: INFERENCE WITH FOMO OLED OVERLAY + WEBSERIAL HOOKS              ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// HEATMAP STREAMING
// Sends max-pooled-across-filters conv2 output as base64 bytes:
//   HEATMAP:<rows>x<cols>:<base64>\n
// ======================================================
void mySendHeatmap() {
  int c2o = myCfg.conv2OutputSize;
  int c2f = myCfg.conv2Filters;
  int cells = c2o * c2o;
  // Allocate tiny raw byte buffer
  uint8_t* myHeatBytes = (uint8_t*)malloc(cells);
  if (!myHeatBytes) return;
  for (int cell = 0; cell < cells; cell++) {
    float mx = 0;
    for (int f = 0; f < c2f; f++) {
      float v = myConv2_output[f * cells + cell];
      if (v > mx) mx = v;
    }
    myHeatBytes[cell] = (uint8_t)(constrain(mx * 255.0f, 0, 255));
  }
  size_t b64Len = 0;
  mbedtls_base64_encode(nullptr, 0, &b64Len, myHeatBytes, cells);
  uint8_t* myB64 = (uint8_t*)malloc(b64Len + 1);
  if (!myB64) { free(myHeatBytes); return; }
  mbedtls_base64_encode(myB64, b64Len, &b64Len, myHeatBytes, cells);
  myB64[b64Len] = 0;
  Serial.printf("HEATMAP:%dx%d:%s\n", c2o, c2o, (char*)myB64);
  free(myHeatBytes);
  free(myB64);
}

void myActionInfer() {
  if (!myWeightsTrained) {
    Serial.println("ERROR: No trained weights! Please train first.");
    u8g2.firstPage();
    do {
      u8g2.setFont(u8g2_font_6x10_tf);
      u8g2.drawStr(0, 12, "No weights!");
      u8g2.drawStr(0, 24, "Train first");
    } while (u8g2.nextPage());
    delay(3000); myResetMenuState(); return;
  }

  Serial.println("\n>>> Inference mode (FOMO WebSerial)");
  Serial.printf("  FOMO grid: %dx%d -> %d cells\n",
                myCfg.fomoGrid, myCfg.fomoGrid, myCfg.fomoCells);
  Serial.println("  T or L = exit  |  HEATMAP_ON / HEATMAP_OFF  |  DETECTION_THRESHOLD:<v>");

  myResetTouchState();

  if (!myInputBuffer || !myDense_output) {
    Serial.println("ERROR: Memory not allocated");
    u8g2.firstPage();
    do { u8g2.drawStr(0, 15, "NOT READY!"); } while (u8g2.nextPage());
    delay(2000); myResetMenuState(); return;
  }

  // Pre-compute resize lookup tables
  int iSz = myCfg.inputSize;
  int* sy_lookup = (int*)malloc(iSz * sizeof(int));
  int* sx_lookup = (int*)malloc(iSz * sizeof(int));
  if (!sy_lookup || !sx_lookup) {
    if (sy_lookup) free(sy_lookup);
    if (sx_lookup) free(sx_lookup);
    Serial.println("ERROR: lookup table alloc failed");
    myResetMenuState(); return;
  }
  for (int i = 0; i < iSz; i++) {
    sy_lookup[i] = min((int)((i+0.5)*240.0/iSz), 239);
    sx_lookup[i] = min((int)((i+0.5)*240.0/iSz), 239);
  }

  int oW = u8g2.getDisplayWidth();
  int oH = u8g2.getDisplayHeight();
  int scX = 240 / oW;
  int scY = 240 / oH;

  unsigned long frameTimes[10] = {};
  int frameIndex = 0;
  int pred       = 0;
  float* predMap = nullptr;
  int peakCell = 0; float peakVal = 0.0f;
  int peakGridX = 0, peakGridY = 0;
  float myActiveThreshold = myCfg.useDynamicThreshold
                            ? myCfg.dynamicThresholdFloor
                            : myCfg.fomoThreshold;

  while (true) {
    unsigned long frameStart = millis();

    // WebSerial command check (handles HEATMAP_ON/OFF, DETECTION_THRESHOLD, exit)
    if (Serial.available()) {
      String line = "";
      unsigned long lineStart = millis();
      while (millis() - lineStart < 30) {
        if (!Serial.available()) { delay(1); continue; }
        char nc = Serial.read();
        if (nc == '\n') break;
        line += nc;
      }
      line.trim();
      if (line.length() > 0) {
        if (line == "t" || line == "T" || line == "l" || line == "L") {
          free(sy_lookup); free(sx_lookup);
          myCamStreaming = false;
          myResetMenuState(); return;
        }
        myHandleStringCommand(line);
      }
    }

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { delay(10); continue; }
    if (!myRgbBuffer) { esp_camera_fb_return(fb); delay(10); continue; }

    if (fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, myRgbBuffer)) {
      myFlipImageHorizontal();

      // Resize 240x240 -> INPUT_SIZE x INPUT_SIZE, normalise
      for (int y = 0; y < iSz; y++) {
        int sy = sy_lookup[y];
        for (int x = 0; x < iSz; x++) {
          int srcIdx = (sy * 240 + sx_lookup[x]) * 3;
          int dstIdx = (y * iSz + x) * 3;
          myInputBuffer[dstIdx]   = myRgbBuffer[srcIdx]   * 0.003921569f;
          myInputBuffer[dstIdx+1] = myRgbBuffer[srcIdx+1] * 0.003921569f;
          myInputBuffer[dstIdx+2] = myRgbBuffer[srcIdx+2] * 0.003921569f;
        }
      }

      myForwardPass(myInputBuffer);

      // Heatmap (if enabled)
      if (myHeatmapEnabled) mySendHeatmap();

      // Predicted class = highest GAP score
      pred = 0;
      for (int i = 1; i < myCfg.numClasses; i++)
        if (myDense_output[i] > myDense_output[pred]) pred = i;

      predMap  = myFomoMap + pred * FOMO_CELLS;
      peakCell = 0; peakVal = predMap[0];
      for (int cell = 1; cell < FOMO_CELLS; cell++)
        if (predMap[cell] > peakVal) { peakVal = predMap[cell]; peakCell = cell; }
      peakGridX = peakCell % FOMO_GRID;
      peakGridY = peakCell / FOMO_GRID;

      myActiveThreshold = myCfg.useDynamicThreshold
        ? max(peakVal * myCfg.dynamicThresholdRatio, myCfg.dynamicThresholdFloor)
        : myCfg.fomoThreshold;

      // ── Compute clusters for serial output ──────────────────────
      struct MySerCluster { int minX, minY, maxX, maxY; float maxVal; };
      MySerCluster mySerClusters[10];
      int mySerClusterCount = 0;
      const int mySerMergeRadius = 6;
      for (int cell = 0; cell < FOMO_CELLS; cell++) {
        if (predMap[cell] < myActiveThreshold) continue;
        int gx = cell % FOMO_GRID, gy = cell / FOMO_GRID;
        int nearest = -1;
        for (int c = 0; c < mySerClusterCount; c++) {
          int cx = (mySerClusters[c].minX + mySerClusters[c].maxX) / 2;
          int cy = (mySerClusters[c].minY + mySerClusters[c].maxY) / 2;
          if (abs(gx-cx) <= mySerMergeRadius && abs(gy-cy) <= mySerMergeRadius) {
            nearest = c; break;
          }
        }
        if (nearest >= 0) {
          mySerClusters[nearest].minX = min(mySerClusters[nearest].minX, gx);
          mySerClusters[nearest].minY = min(mySerClusters[nearest].minY, gy);
          mySerClusters[nearest].maxX = max(mySerClusters[nearest].maxX, gx);
          mySerClusters[nearest].maxY = max(mySerClusters[nearest].maxY, gy);
          mySerClusters[nearest].maxVal = max(mySerClusters[nearest].maxVal, predMap[cell]);
        } else if (mySerClusterCount < 10) {
          mySerClusters[mySerClusterCount++] = {gx, gy, gx, gy, predMap[cell]};
        }
      }

      // ── INFER: serial token (parsed by index.html JS) ────────────
      // Format: INFER:<className>,<gapConf>,<peakX>,<peakY>,<peakVal>,<numClusters>
      Serial.printf("INFER:%s,%.4f,%d,%d,%.4f,%d\n",
                    myClassLabels[pred].c_str(), myDense_output[pred],
                    peakGridX, peakGridY, peakVal, mySerClusterCount);

      // ── Per-frame serial summary ─────────────────────────────────
      frameTimes[frameIndex] = millis() - frameStart;
      float fps = 1000.0f / max((unsigned long)1, frameTimes[frameIndex]);
      Serial.printf("F%d: %lums (%.1fFPS) pred=%s(%.0f%%) peak=(%d,%d)@%.2f thr=%.2f",
                    frameIndex+1, frameTimes[frameIndex], fps,
                    myClassLabels[pred].c_str(), myDense_output[pred]*100,
                    peakGridX, peakGridY, peakVal, myActiveThreshold);
      Serial.print(" | ");
      for (int i = 0; i < myCfg.numClasses; i++)
        Serial.printf(" %s=%.0f%%", myClassLabels[i].c_str(), myDense_output[i]*100);
      Serial.printf(" | Clusters(%d):", mySerClusterCount);
      for (int c = 0; c < mySerClusterCount; c++) {
        int cx = (mySerClusters[c].minX + mySerClusters[c].maxX) / 2;
        int cy = (mySerClusters[c].minY + mySerClusters[c].maxY) / 2;
        Serial.printf(" [%d,%d]@%.2f", cx, cy, mySerClusters[c].maxVal);
      }
      if (mySerClusterCount == 0) Serial.print(" none");
      Serial.println();

      // ── OLED every 10 frames ──────────────────────────────────────
      if (frameIndex == 9) {
        u8g2.firstPage();
        do {
          for (int ox = 0; ox < oW; ox++) {
            for (int oy = 0; oy < oH; oy++) {
              int pi = ((oy * scY) * 240 + (ox * scX)) * 3;
              uint8_t bright = (myRgbBuffer[pi] + myRgbBuffer[pi+1] + myRgbBuffer[pi+2]) / 3;
              if (bright > 100) u8g2.drawPixel(ox, oy);
            }
          }
          struct MyCluster { int minX, minY, maxX, maxY; float maxVal; };
          MyCluster myClusters[10];
          int myClusterCount = 0;
          const int myMergeRadius = 6;
          for (int cell = 0; cell < FOMO_CELLS; cell++) {
            if (predMap[cell] < myActiveThreshold) continue;
            int gx = cell % FOMO_GRID, gy = cell / FOMO_GRID;
            int nearest = -1;
            for (int c = 0; c < myClusterCount; c++) {
              int cx = (myClusters[c].minX + myClusters[c].maxX) / 2;
              int cy = (myClusters[c].minY + myClusters[c].maxY) / 2;
              if (abs(gx-cx) <= myMergeRadius && abs(gy-cy) <= myMergeRadius) { nearest=c; break; }
            }
            if (nearest >= 0) {
              myClusters[nearest].minX = min(myClusters[nearest].minX, gx);
              myClusters[nearest].minY = min(myClusters[nearest].minY, gy);
              myClusters[nearest].maxX = max(myClusters[nearest].maxX, gx);
              myClusters[nearest].maxY = max(myClusters[nearest].maxY, gy);
              myClusters[nearest].maxVal = max(myClusters[nearest].maxVal, predMap[cell]);
            } else if (myClusterCount < 10) {
              myClusters[myClusterCount++] = {gx, gy, gx, gy, predMap[cell]};
            }
          }
          for (int c = 0; c < myClusterCount; c++) {
            int rx1 = (int)(myClusters[c].minX * oW / FOMO_GRID);
            int ry1 = (int)(myClusters[c].minY * oH / FOMO_GRID);
            int rx2 = (int)((myClusters[c].maxX + 1) * oW / FOMO_GRID);
            int ry2 = (int)((myClusters[c].maxY + 1) * oH / FOMO_GRID);
            u8g2.setColorIndex(0); u8g2.drawFrame(rx1+1, ry1+1, rx2-rx1-1, ry2-ry1-1);
            u8g2.setColorIndex(1); u8g2.drawFrame(rx1,   ry1,   rx2-rx1,   ry2-ry1);
          }
        } while (u8g2.nextPage());
      }

      // Camera stream send (for live preview in browser)
      if (myCamStreaming) myBase64SendFrame(fb);
    }

    esp_camera_fb_return(fb);
    frameIndex++;
    if (frameIndex >= 10) {
      int touchVal = myReadTouch();
      if (touchVal > myThresholdPress) {
        Serial.println("Touch detected — exiting inference");
        delay(200);
        free(sy_lookup); free(sx_lookup);
        myCamStreaming = false;
        myResetMenuState(); return;
      }
      frameIndex = 0;
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 4: MENU SYSTEM                                                      ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

void myResetMenuState() {
  myIsSelected = false;
  myResetTouchState();
  myLastActivityTime = millis();
  myDrawMenu();
}

void myDrawMenu() {
  Serial.println("\n=== MENU ===");
  for (int i = 1; i <= myTotalItems; i++) {
    String label =
      (i <= myCfg.numClasses) ? myClassLabels[i-1] :
      (i == myCfg.numClasses+1) ? "Train" : "Infer";
    if (i == myMenuIndex) Serial.print(" > ");
    else                  Serial.print("   ");
    Serial.printf("%d. %s\n", i, label.c_str());
  }
  Serial.println("Commands: t=next  l=select  1-9=direct  STATUS  SD_LIST:/  CAM_CAPTURE");

  u8g2.firstPage();
  do {
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0, 8, "TAP:Next HOLD:Ok");
    int myStartItem = (myMenuIndex <= myCfg.numClasses) ? 1 : myMenuIndex - 2;
    for (int i = 0; i < 3; i++) {
      int cur = myStartItem + i;
      if (cur > myTotalItems) break;
      String label =
        (cur <= myCfg.numClasses) ? myClassLabels[cur-1] :
        (cur == myCfg.numClasses+1) ? "Train" : "Infer";
      int y = 18 + i * 9;
      if (cur == myMenuIndex) u8g2.drawStr(0, y, ("> " + label).c_str());
      else                    u8g2.drawStr(0, y, ("  " + label).c_str());
    }
  } while (u8g2.nextPage());
}

// v76: centralised dispatch by menu index
void myDispatchByIndex(int idx) {
  if (idx <= myCfg.numClasses) myActionCollect(idx - 1);
  else if (idx == myCfg.numClasses + 1) myActionTrain();
  else                                   myActionInfer();
}

void myHandleMenuNavigation() {
  unsigned long myCurrentMillis = millis();

  if (!myIsSelected && Serial.available()) {
    // Accumulate a full line for string commands
    char c = Serial.peek();
    // Single-char menu shortcuts
    if (c >= '1' && c <= '9') {
      Serial.read();
      int newIndex = c - '0';
      if (newIndex <= myTotalItems) {
        myMenuIndex = newIndex; myIsSelected = true;
        myLastActivityTime = myCurrentMillis;
        myDispatchByIndex(myMenuIndex);
      }
      return;
    }
    if (c == 't' || c == 'T') {
      Serial.read();
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
      return;
    }
    if (c == 'l' || c == 'L') {
      Serial.read();
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myDispatchByIndex(myMenuIndex);
      return;
    }
    // Otherwise: read a full line for string command
    String line = "";
    unsigned long lineStart = millis();
    while (millis() - lineStart < 100) {
      if (!Serial.available()) { delay(1); continue; }
      char nc = Serial.read();
      if (nc == '\n') break;
      if (nc != '\r') line += nc;
    }
    line.trim();
    if (line.length() > 0) myHandleStringCommand(line);
  }

  if (!myIsSelected) {
    int touchAction = myCheckTouchInput();
    if (touchAction == 1) {
      if (myCurrentMillis - myLastTapTime > myTapCooldown) {
        myMenuIndex++;
        if (myMenuIndex > myTotalItems) myMenuIndex = 1;
        myDrawMenu();
        myLastTapTime = myCurrentMillis;
        myLastActivityTime = myCurrentMillis;
      }
    } else if (touchAction == 2) {
      myIsSelected = true;
      myLastActivityTime = myCurrentMillis;
      myDispatchByIndex(myMenuIndex);
    }
  }
}


// ██████████████████████████████████████████████████████████████████████████████
// ██                                                                          ██
// ██  PART 5: WEBSERIAL STRING COMMAND HANDLER                                 ██
// ██                                                                          ██
// ██  Handles: STATUS, SD browser, camera, binary transfer, FOMO commands     ██
// ██                                                                          ██
// ██████████████████████████████████████████████████████████████████████████████

// ======================================================
// CAMERA HELPERS
// ======================================================
void myBase64SendFrame(camera_fb_t* fb) {
  size_t b64Len = 0;
  mbedtls_base64_encode(nullptr, 0, &b64Len, fb->buf, fb->len);
  uint8_t* myB64 = (uint8_t*)malloc(b64Len + 1);
  if (!myB64) return;
  mbedtls_base64_encode(myB64, b64Len, &b64Len, fb->buf, fb->len);
  myB64[b64Len] = 0;
  Serial.printf("FRAME_B64:%s\n", (char*)myB64);
  free(myB64);
}

void myCamCaptureSend() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return;
  myBase64SendFrame(fb);
  esp_camera_fb_return(fb);
}

void myActionWebStream() {
  Serial.println("WebStream: press T/L or touch to stop");
  while (true) {
    myCamCaptureSend();
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 't' || c == 'T' || c == 'l' || c == 'L') break;
    }
    if (myCheckTouchInput() > 0) break;
    delay(200);
  }
  myResetMenuState();
}

// ======================================================
// SD HELPERS
// ======================================================
String myNormPath(String p) {
  p.trim();
  if (!p.startsWith("/")) p = "/" + p;
  return p;
}

void mySdListDir(const String& path) {
  if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
  File dir = SD.open(path);
  if (!dir || !dir.isDirectory()) { Serial.println("ERR:Cannot open dir"); return; }
  Serial.printf("SD_LIST_START:%s\n", path.c_str());
  while (File entry = dir.openNextFile()) {
    if (entry.isDirectory())
      Serial.printf("DIR:%s\n", entry.name());
    else
      Serial.printf("FILE:%s:%d\n", entry.name(), (int)entry.size());
    entry.close();
  }
  Serial.println("SD_LIST_END");
  dir.close();
}

void mySdReadText(const String& path) {
  if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
  File f = SD.open(path);
  if (!f) { Serial.println("ERR:Cannot open file"); return; }
  Serial.printf("SD_READ_START:%s\n", path.c_str());
  while (f.available()) Serial.write(f.read());
  Serial.println("\nSD_READ_END");
  f.close();
}

void mySdReadJpeg(const String& path) {
  if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
  File f = SD.open(path);
  if (!f) { Serial.println("ERR:Cannot open file"); return; }
  size_t sz = f.size();
  uint8_t* buf = (uint8_t*)ps_malloc(sz);
  if (!buf) { f.close(); Serial.println("ERR:OOM"); return; }
  f.read(buf, sz);
  f.close();
  size_t b64Len = 0;
  mbedtls_base64_encode(nullptr, 0, &b64Len, buf, sz);
  uint8_t* myB64 = (uint8_t*)malloc(b64Len + 1);
  if (!myB64) { free(buf); Serial.println("ERR:OOM b64"); return; }
  mbedtls_base64_encode(myB64, b64Len, &b64Len, buf, sz);
  myB64[b64Len] = 0;
  free(buf);
  Serial.printf("SD_JPEG:%s:%s\n", path.c_str(), (char*)myB64);
  free(myB64);
}

void mySdReadBinaryHead(const String& path, uint16_t numBytes) {
  if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
  File f = SD.open(path);
  if (!f) { Serial.println("ERR:Cannot open file"); return; }
  Serial.printf("SD_HEAD_START:%s:%d\n", path.c_str(), numBytes);
  int count = 0;
  while (f.available() && count < numBytes) { Serial.write(f.read()); count++; }
  Serial.println("\nSD_HEAD_END");
  f.close();
}

bool mySdRemoveDirRecursive(const String& path, int& deleted) {
  if (!mySDavailable) return false;
  File dir = SD.open(path);
  if (!dir) return false;
  while (File entry = dir.openNextFile()) {
    String entPath = String(entry.path());
    if (entry.isDirectory()) { entry.close(); mySdRemoveDirRecursive(entPath, deleted); }
    else { entry.close(); if (SD.remove(entPath)) deleted++; }
  }
  dir.close();
  return SD.rmdir(path);
}

// ======================================================
// MAIN STRING COMMAND DISPATCHER
// ======================================================
void myHandleStringCommand(const String& cmd) {

  // ── STATUS ─────────────────────────────────────────────────────────────────
  if (cmd == "STATUS") {
    Serial.printf("STATUS:esp32-fomo-webserial-%d\n", CURRENT_VERSION);
    Serial.printf("STATUS:numClasses=%d inputSize=%d fomoGrid=%d fomoCells=%d\n",
                  myCfg.numClasses, myCfg.inputSize, myCfg.fomoGrid, myCfg.fomoCells);
    Serial.printf("STATUS:sd=%s weights=%s heatmap=%s camStream=%s\n",
                  mySDavailable ? "ok" : "none",
                  myWeightsTrained ? "trained" : "untrained",
                  myHeatmapEnabled ? "on" : "off",
                  myCamStreaming ? "on" : "off");
    Serial.println("STATUS:commands=STATUS,SD_LIST,SD_READ,SD_WRITE,SD_JPEG,SD_HEAD,"
                   "SD_DELETE,SD_RMDIR,SD_MKDIR,SD_JPEG_WRITE_START,SD_JPEG_CHUNK,"
                   "SD_JPEG_WRITE_END,SD_TEXT_WRITE_START,SD_TEXT_CHUNK,SD_TEXT_WRITE_END,"
                   "FILE_SEND_START,FILE_CHUNK,FILE_SEND_END,"
                   "CAM_CAPTURE,CAM_STREAM,CAM_STREAM_STOP,HEATMAP_ON,HEATMAP_OFF,"
                   "HEATMAP_STATUS,DETECTION_THRESHOLD,COLLECT_COUNT,INFER,TRAIN_EPOCH");
    return;
  }

  // ── HEATMAP ────────────────────────────────────────────────────────────────
  if (cmd == "HEATMAP_ON")  { myHeatmapEnabled = true;  Serial.println("OK:HEATMAP_ON");  return; }
  if (cmd == "HEATMAP_OFF") { myHeatmapEnabled = false; Serial.println("OK:HEATMAP_OFF"); return; }
  if (cmd == "HEATMAP_STATUS") {
    Serial.printf("HEATMAP_STATUS:%s %dx%d\n",
                  myHeatmapEnabled ? "on" : "off",
                  myCfg.conv2OutputSize, myCfg.conv2OutputSize);
    return;
  }

  // ── FOMO: DETECTION_THRESHOLD ───────────────────────────────────────────────
  if (cmd.startsWith("DETECTION_THRESHOLD:")) {
    float val = cmd.substring(20).toFloat();
    myCfg.fomoThreshold         = val;
    myCfg.dynamicThresholdFloor = val;
    Serial.printf("OK:DETECTION_THRESHOLD:%.3f\n", val);
    return;
  }

  // ── CAMERA ─────────────────────────────────────────────────────────────────
  if (cmd == "CAM_CAPTURE") { myCamCaptureSend(); return; }
  if (cmd == "CAM_STREAM")  { myCamStreaming = true;  Serial.println("OK:CAM_STREAM_START"); return; }
  if (cmd == "CAM_STREAM_STOP") { myCamStreaming = false; Serial.println("OK:CAM_STREAM_STOP"); return; }

  // ── SD BROWSER ─────────────────────────────────────────────────────────────
  if (cmd.startsWith("SD_LIST:"))  { if (!mySDavailable) { Serial.println("ERR:SD not available"); return; } mySdListDir(myNormPath(cmd.substring(8)));  return; }
  if (cmd.startsWith("SD_READ:"))  { if (!mySDavailable) { Serial.println("ERR:SD not available"); return; } mySdReadText(myNormPath(cmd.substring(8)));  return; }
  if (cmd.startsWith("SD_JPEG:"))  { if (!mySDavailable) { Serial.println("ERR:SD not available"); return; } mySdReadJpeg(myNormPath(cmd.substring(8)));  return; }
  if (cmd.startsWith("SD_HEAD:"))  {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    String rest = cmd.substring(8);
    int colon = rest.lastIndexOf(':');
    if (colon < 0) { mySdReadBinaryHead(myNormPath(rest), 256); }
    else           { mySdReadBinaryHead(myNormPath(rest.substring(0, colon)), rest.substring(colon+1).toInt()); }
    return;
  }
  if (cmd.startsWith("SD_DELETE:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    String p = myNormPath(cmd.substring(10));
    if (SD.remove(p)) Serial.printf("OK:SD_DELETE:%s\n", p.c_str());
    else              Serial.printf("ERR:SD_DELETE:%s\n", p.c_str());
    return;
  }
  if (cmd.startsWith("SD_RMDIR:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    String p = myNormPath(cmd.substring(9));
    int deleted = 0;
    if (mySdRemoveDirRecursive(p, deleted)) Serial.printf("OK:SD_RMDIR:%s deleted=%d\n", p.c_str(), deleted);
    else                                    Serial.printf("ERR:SD_RMDIR:%s\n", p.c_str());
    return;
  }
  if (cmd.startsWith("SD_MKDIR:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    String p = myNormPath(cmd.substring(9));
    if (SD.mkdir(p)) Serial.printf("OK:SD_MKDIR:%s\n", p.c_str());
    else             Serial.printf("ERR:SD_MKDIR:%s\n", p.c_str());
    return;
  }

  // ── SD_WRITE (legacy short-content write) ──────────────────────────────────
  if (cmd.startsWith("SD_WRITE:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    String rest = cmd.substring(9);
    int colon = rest.indexOf(':');
    if (colon < 0) { Serial.println("ERR:SD_WRITE bad format"); return; }
    String path    = myNormPath(rest.substring(0, colon));
    String content = rest.substring(colon + 1);
    content.replace("\\n", "\n");
    myEnsureParentDir(path);
    File f = SD.open(path, FILE_WRITE);
    if (f) { f.print(content); f.close(); Serial.printf("OK:SD_WRITE:%s\n", path.c_str()); }
    else     Serial.printf("ERR:SD_WRITE:%s\n", path.c_str());
    return;
  }

  // ── SD_TEXT_WRITE_START / CHUNK / END ──────────────────────────────────────
  if (cmd.startsWith("SD_TEXT_WRITE_START:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    myTextWritePath    = myNormPath(cmd.substring(20));
    myTextWriteContent = "";
    myTextWriteActive  = true;
    Serial.printf("OK:SD_TEXT_START:%s\n", myTextWritePath.c_str());
    return;
  }
  if (cmd.startsWith("SD_TEXT_CHUNK:") && myTextWriteActive) {
    String chunk = cmd.substring(14);
    chunk.replace("\\n", "\n");
    myTextWriteContent += chunk;
    Serial.println("OK:SD_TEXT_CHUNK");
    return;
  }
  if (cmd == "SD_TEXT_WRITE_END" && myTextWriteActive) {
    myTextWriteActive = false;
    myEnsureParentDir(myTextWritePath);
    File f = SD.open(myTextWritePath, FILE_WRITE);
    if (f) {
      f.print(myTextWriteContent);
      f.close();
      Serial.printf("OK:SD_TEXT_END:%s bytes=%d\n", myTextWritePath.c_str(), myTextWriteContent.length());
    } else {
      Serial.printf("ERR:SD_TEXT_END:%s\n", myTextWritePath.c_str());
    }
    myTextWriteContent = "";
    return;
  }

  // ── SD_JPEG_WRITE_START / CHUNK / END ──────────────────────────────────────
  if (cmd.startsWith("SD_JPEG_WRITE_START:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    myJpegWritePath   = myNormPath(cmd.substring(20));
    myJpegWriteB64    = "";
    myJpegWriteActive = true;
    Serial.printf("OK:SD_JPEG_START:%s\n", myJpegWritePath.c_str());
    return;
  }
  if (cmd.startsWith("SD_JPEG_CHUNK:") && myJpegWriteActive) {
    myJpegWriteB64 += cmd.substring(14);
    Serial.println("OK:SD_JPEG_CHUNK");
    return;
  }
  if (cmd == "SD_JPEG_WRITE_END" && myJpegWriteActive) {
    myJpegWriteActive = false;
    size_t b64Len = myJpegWriteB64.length();
    size_t binLen = 0;
    uint8_t* jpegBuf = (uint8_t*)ps_malloc(b64Len);
    if (!jpegBuf) { Serial.println("ERR:OOM SD_JPEG_WRITE_END"); myJpegWriteB64 = ""; return; }
    mbedtls_base64_decode(jpegBuf, b64Len, &binLen,
                          (const uint8_t*)myJpegWriteB64.c_str(), b64Len);
    myJpegWriteB64 = "";
    myEnsureParentDir(myJpegWritePath);
    File f = SD.open(myJpegWritePath, FILE_WRITE);
    if (f) {
      f.write(jpegBuf, binLen);
      f.close();
      Serial.printf("OK:SD_JPEG_END:%s bytes=%d\n", myJpegWritePath.c_str(), (int)binLen);
    } else {
      Serial.printf("ERR:SD_JPEG_END:%s\n", myJpegWritePath.c_str());
    }
    free(jpegBuf);
    return;
  }

  // ── FILE_SEND_START / FILE_CHUNK / FILE_SEND_END  (v78 binary protocol) ────
  // Used by torchjs to push myWeights.bin and config.json to the ESP32 SD card.
  // FILE_SEND_START:/path/to/file:totalBytes
  // FILE_CHUNK:<chunkIndex>:<xorChecksum>:<hexEncodedBytes>
  // FILE_SEND_END:<totalBytesExpected>
  if (cmd.startsWith("FILE_SEND_START:")) {
    if (!mySDavailable) { Serial.println("ERR:SD not available"); return; }
    // Parse path and size
    String rest = cmd.substring(16);
    int lastColon = rest.lastIndexOf(':');
    if (lastColon < 0) { Serial.println("ERR:FILE_SEND_START bad format"); return; }
    myFileRecvPath  = myNormPath(rest.substring(0, lastColon));
    myFileRecvBytes = rest.substring(lastColon + 1).toInt();
    myEnsureParentDir(myFileRecvPath);
    if (myFileRecvHandle) myFileRecvHandle.close();
    myFileRecvHandle = SD.open(myFileRecvPath, FILE_WRITE);
    if (!myFileRecvHandle) { Serial.printf("ERR:FILE_SEND_START:%s\n", myFileRecvPath.c_str()); return; }
    myFileRecvActive = true;
    Serial.printf("OK:FILE_READY:%s:%d\n", myFileRecvPath.c_str(), myFileRecvBytes);
    return;
  }
  if (cmd.startsWith("FILE_CHUNK:") && myFileRecvActive) {
    // Format: FILE_CHUNK:<idx>:<xorChecksum>:<hexData>
    String rest = cmd.substring(11);
    int c1 = rest.indexOf(':');
    int c2 = rest.indexOf(':', c1 + 1);
    if (c1 < 0 || c2 < 0) { Serial.println("ERR:CHUNK_FORMAT"); return; }
    int chunkIdx    = rest.substring(0, c1).toInt();
    int xorExpected = rest.substring(c1 + 1, c2).toInt();
    String hexData  = rest.substring(c2 + 1);
    int hexLen = hexData.length();
    if (hexLen % 2 != 0) { Serial.printf("ERR:CHUNK_RETRY:%d:odd_hex\n", chunkIdx); return; }
    int binLen = hexLen / 2;
    uint8_t* buf = (uint8_t*)malloc(binLen);
    if (!buf) { Serial.printf("ERR:CHUNK_RETRY:%d:OOM\n", chunkIdx); return; }
    int xorActual = 0;
    for (int i = 0; i < binLen; i++) {
      char hi = hexData[i*2], lo = hexData[i*2+1];
      auto hexNib = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
      };
      buf[i] = (hexNib(hi) << 4) | hexNib(lo);
      xorActual ^= buf[i];
    }
    if (xorActual != xorExpected) {
      free(buf);
      Serial.printf("ERR:CHUNK_RETRY:%d:checksum_mismatch\n", chunkIdx); return;
    }
    myFileRecvHandle.write(buf, binLen);
    free(buf);
    Serial.printf("OK:CHUNK_ACK:%d\n", chunkIdx);
    return;
  }
  if (cmd.startsWith("FILE_SEND_END:") && myFileRecvActive) {
    int expectedTotal = cmd.substring(14).toInt();
    int actualSize = (int)myFileRecvHandle.size();
    myFileRecvHandle.close();
    myFileRecvActive = false;
    if (actualSize == expectedTotal) {
      Serial.printf("OK:FILE_DONE:%s:%d\n", myFileRecvPath.c_str(), actualSize);
      // If this was the weights file, reload weights and config
      if (myFileRecvPath.endsWith(".bin")) {
        myLoadConfig();
        mySyncLegacyVars();
        myAllocateMemory(true);
        myLoadWeights(true);
      } else if (myFileRecvPath.endsWith("config.json")) {
        myLoadConfig();
        mySyncLegacyVars();
        myAllocateMemory(true);
      }
    } else {
      Serial.printf("ERR:FILE_SIZE_MISMATCH:%d!=%d\n", actualSize, expectedTotal);
    }
    return;
  }

  // Unrecognised command
  Serial.printf("ERR:UNKNOWN_CMD:%s\n", cmd.c_str());
}
