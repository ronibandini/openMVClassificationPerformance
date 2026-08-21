# 👁️⚡ OpenMV RT1062 Classification Performance

**A compact benchmark project for measuring image-classification inference performance and power consumption on the OpenMV Cam RT1062 using Edge Impulse and MicroPython.**

This project evaluates how the **OpenMV Cam RT1062** behaves when running an Edge Impulse image-classification model at different camera resolutions.

The goal is not to solve a difficult classification problem. Instead, the model deliberately distinguishes between two visually different objects so the experiment can focus on:

- ⚡ inference time
- 🎥 camera resolution
- 📈 frames per second
- 🔋 power consumption
- 🐍 MicroPython development workflow
- 🧠 Edge Impulse deployment
- 📷 practical usability of the OpenMV Cam RT1062 for edge AI

The reference model classifies:

```text
LEGO figure
vs.
small blue ball
```

using a lightweight:

```text
96 × 96
```

image-classification impulse.

---

## ✨ Features

- 👁️ Embedded image classification
- ⚡ OpenMV Cam RT1062
- 🧠 Edge Impulse model deployment
- 🐍 MicroPython
- 🎥 Multiple camera-resolution tests
- ⏱️ Inference-time measurement
- 📈 FPS monitoring
- 🔋 Idle vs. inference power measurements
- 📊 Spreadsheet with benchmark data
- 💾 Standalone execution from OpenMV flash
- 📷 5 MP OV5640 camera
- 🌐 Wi-Fi, Bluetooth and Ethernet available on the board
- 💽 microSD support
- 🖥️ OpenMV IDE workflow
- 📜 MIT licensed

---

## 🎯 Project goal

Machine Learning benchmarks often concentrate only on:

```text
model accuracy
```

but an embedded vision project also depends on:

```text
inference latency
+
camera resolution
+
memory
+
power consumption
+
development complexity
```

This project investigates those practical factors on the OpenMV Cam RT1062.

```text
Edge Impulse model
        │
        ▼
OpenMV Cam RT1062
        │
        ├── capture frame
        ├── preprocess image
        ├── run classifier
        ├── measure inference time
        ├── calculate FPS
        └── measure current draw
```

---

## 🧠 Classification model

The benchmark uses a deliberately simple two-class classification problem.

The dataset contains:

```text
30 images → LEGO figure
30 images → blue ball
```

for a total of:

```text
60 images
```

The two objects are visually distinct, which makes the classification task intentionally easy.

The purpose is to reduce model ambiguity and concentrate on the performance of the hardware and deployment workflow.

---

## 📐 Edge Impulse configuration

The reference impulse uses:

```text
Image width:     96
Image height:    96
Learning block:  Classification
```

Training settings:

```text
Training cycles: 10
Learning rate:   0.0005
```

The resulting model is exported using:

```text
OpenMV Library
```

from the Edge Impulse Deployment page.

---

## 🔄 Training workflow

```text
Photograph LEGO figure
        │
        ├── 30 images
        │
Photograph blue ball
        │
        └── 30 images
               │
               ▼
        Edge Impulse
               │
               ▼
         Create Impulse
               │
               ▼
          96 × 96 image
               │
               ▼
         Classification
               │
               ▼
             Train
               │
               ▼
         Test model
               │
               ▼
       OpenMV deployment
```

---

# 📷 OpenMV Cam RT1062

The project runs on the **OpenMV Cam RT1062**, a microcontroller-based machine-vision board centered around the NXP i.MX RT1062.

Current hardware highlights include:

- ARM Cortex-M7 at 600 MHz
- double-precision FPU
- 32 MB external SDRAM
- 1 MB internal SRAM
- 16 MB QSPI Flash
- OV5640 5 MP camera sensor
- microSD slot
- Wi-Fi
- Bluetooth 5.1
- 10/100 Ethernet
- RTC
- accelerometer
- USB-C
- LiPo battery support
- low-power sleep modes
- programmable GPIO

Official product page:

**[OpenMV Cam RT1062](https://openmv.io/products/openmv-cam-rt)**

Documentation:

**[OpenMV Cam RT1062 documentation](https://docs.openmv.io/dev/openmvcam/quickref/openmv-rt1062.html)**

---

## 🧠 NXP i.MX RT1062

The RT1062 combines microcontroller-style operation with considerably more performance than smaller embedded vision boards.

Its:

```text
600 MHz Cortex-M7
+
large external SDRAM
+
camera interface
```

makes it suitable for:

- embedded image classification
- FOMO object detection
- traditional computer vision
- camera-triggered automation
- robotics
- low-power remote vision
- industrial prototyping

while still being programmable using **MicroPython**.

---

# 🐍 MicroPython

OpenMV applications are normally written in MicroPython.

This makes it possible to interact with:

```text
camera
GPIO
files
network
ML models
image processing
```

using Python-like syntax directly on the embedded board.

A typical vision loop follows this structure:

```python
while True:
    img = sensor.snapshot()

    # Run inference

    # Print results
```

The benchmark script extends the standard Edge Impulse classification example to collect performance information.

---

# 📄 Main benchmark script

The repository includes:

```text
ei_classification_performance.py
```

This script is based on the image-classification example generated by Edge Impulse for OpenMV, with additional logic for evaluating runtime performance.

Its purpose is to measure values such as:

```text
inference time
FPS
classification results
```

while changing the source-image resolution.

---

# 🎥 Camera resolution

The standard Edge Impulse OpenMV classification example uses a relatively small source frame.

The project starts with:

```text
240 × 240
```

and then tests larger source resolutions.

To reach higher resolutions, the camera frame size is switched to:

```python
sensor.set_framesize(sensor.SXGA)
```

This allows benchmark experiments up to approximately:

```text
1024 × 1024
```

for the classification workflow used in the project.

---

## 🧩 Model input vs. camera resolution

The Edge Impulse model itself remains:

```text
96 × 96
```

The variable being changed is the **camera source resolution**.

Conceptually:

```text
Camera
240 × 240
     │
     ▼
preprocessing
     │
     ▼
96 × 96 model
```

or:

```text
Camera
1024 × 1024
     │
     ▼
preprocessing
     │
     ▼
96 × 96 model
```

This makes it possible to investigate the cost of processing a larger source image without changing the neural-network architecture.

---

# ⏱️ Inference performance

The benchmark records the amount of time required for the classification loop.

The general behavior observed in the project is:

```text
larger camera frame
       │
       ▼
more image processing
       │
       ▼
greater inference / pipeline time
```

while the underlying classifier remains the same.

The exact results collected during the experiment are stored in:

[`OpenMV RT1062 Edge Impulse.xlsx`](OpenMV%20RT1062%20Edge%20Impulse.xlsx)

---

# 📊 Benchmark spreadsheet

The repository includes:

```text
OpenMV RT1062 Edge Impulse.xlsx
```

for recording the hardware measurements obtained during testing.

The workbook is useful for comparing:

```text
resolution
inference time
FPS
idle consumption
inference consumption
```

across the different configurations.

Keeping the measurements outside the firmware also makes it easier to:

- graph results
- compare configurations
- repeat tests
- add new model versions
- benchmark future firmware releases

---

# 🔋 Power consumption

Performance is only one side of an embedded-AI benchmark.

The project also measures current draw during:

```text
idle
```

and:

```text
ML inference
```

to investigate whether higher-resolution image processing produces a meaningful increase in power consumption.

---

## ⚡ Measurement setup

To measure current directly, headers were added to:

```text
VIN
GND
```

and a multimeter was placed in series with the power path.

```text
Power supply
     │
     ▼
 Multimeter
     │
     ▼
 VIN
     │
OpenMV RT1062
     │
    GND
```

This prevents the USB connection from bypassing the current-measurement path.

---

## 📏 Measurement precision

The reference experiment used the current-measurement mode of a general-purpose multimeter.

The results are suitable for comparative experimentation, but very precise power characterization should use dedicated equipment such as:

- precision ammeter
- Joulescope
- power analyzer
- source measurement unit
- high-resolution current monitor

The most useful result from this benchmark is therefore the **relative behavior across configurations**, rather than treating every current reading as laboratory-grade characterization.

---

# 💾 Standalone execution

Power measurements require the camera to operate without the normal USB programming connection.

OpenMV IDE supports saving a script directly to the board.

Use:

```text
Tools
→ Save Open Script to OpenMV Cam
```

The script can then be executed after reset or power-up.

This allows:

```text
external power
     │
     ▼
OpenMV boot
     │
     ▼
main.py
     │
     ▼
classification benchmark
```

without a development computer attached.

---

# 🖥️ OpenMV IDE

Download:

**[OpenMV IDE](https://openmv.io/pages/download)**

The IDE provides:

- MicroPython editor
- serial console
- live camera framebuffer
- script execution
- firmware update
- onboard filesystem access
- examples
- model-development workflow

After opening the benchmark script, connect to the camera and run it.

The live framebuffer can be used to confirm:

- camera focus
- framing
- model input
- classification target

while the terminal displays performance information.

---

# 🧠 Edge Impulse deployment

After training the model:

```text
Edge Impulse
→ Deployment
→ OpenMV Library
→ Build
```

Extract the generated archive.

For a small classification model, the important files include:

```text
trained.tflite
labels.txt
```

Copy these to the OpenMV mass-storage volume exposed over USB.

```text
OpenMV Cam/
├── trained.tflite
├── labels.txt
└── main.py / benchmark script
```

---

# 🧠 OpenMV Library vs. OpenMV Firmware

Current Edge Impulse documentation supports two deployment methods.

## OpenMV Library

Best suited to smaller models.

```text
trained.tflite
+
labels.txt
+
MicroPython script
```

are copied to the camera filesystem.

This is the approach used by the original benchmark.

---

## OpenMV Firmware

Edge Impulse also provides a firmware deployment method where the model is compiled directly into a custom OpenMV firmware image.

Current Edge Impulse documentation describes firmware deployment as the preferred approach for larger or production-oriented projects.

The RT1062 target currently appears as:

```text
edge_impulse_firmware_openmv_rt1060.bin
```

despite the hardware being named RT1062.

---

# 🔢 Quantized models

Current Edge Impulse OpenMV deployment supports:

```text
int8 quantized models
```

If a model fails to build or deploy, verify that the model is quantized.

Quantization helps reduce:

- RAM
- Flash usage
- model size
- computational cost

which is especially important on microcontrollers.

---

# 🚥 OpenMV status LED

The OpenMV Cam exposes board-state information through its onboard status LED.

The project documentation uses these states:

| Color | Meaning |
|---|---|
| Green | Bootloader running |
| Blue | `main.py` executing |
| White | Firmware panic / hardware failure |

These indicators can be useful when the camera is operating without an attached computer.

---

# 🛠️ Firmware recovery

If a firmware update leaves the RT1062 unable to boot normally, the board can be forced into its serial bootloader using the documented recovery procedure.

The original project uses a temporary connection between:

```text
SBL
and
3.3 V
```

before reflashing the board from OpenMV IDE.

Always check the latest OpenMV hardware documentation before performing a recovery procedure, since board revisions and bootloader tooling can change.

---

# 🌐 Connectivity

Although this benchmark runs locally, the RT1062 also provides:

```text
Wi-Fi
Bluetooth
Ethernet
```

which means a derived project could report classifications or performance statistics remotely.

Possible architectures include:

```text
OpenMV
  │
  ├── MQTT
  ├── HTTP
  ├── REST API
  ├── local socket
  └── telemetry server
```

without requiring an additional host computer.

---

# 💽 microSD

The RT1062 includes a microSD slot.

This is particularly useful for Machine Learning experiments because the same device can potentially:

```text
capture dataset
      │
      ▼
save images
      │
      ▼
inspect / upload data
      │
      ▼
run inference
```

A camera can therefore participate in both:

```text
data acquisition
```

and:

```text
model deployment
```

workflows.

---

# 🧪 Reproducing the benchmark

## 1. Clone the repository

```bash
git clone https://github.com/ronibandini/openMVClassificationPerformance.git
cd openMVClassificationPerformance
```

---

## 2. Install OpenMV IDE

Download the current version:

**[OpenMV IDE](https://openmv.io/pages/download)**

---

## 3. Create an Edge Impulse project

Create a new project at:

**[Edge Impulse Studio](https://studio.edgeimpulse.com/)**

---

## 4. Collect two image classes

The reference experiment uses:

```text
30 LEGO figure images
30 blue ball images
```

You can use any two visually distinct objects.

For a benchmark, simple classes are useful because the purpose is hardware performance rather than maximizing classifier complexity.

---

## 5. Create the impulse

Configure:

```text
Image width:  96
Image height: 96
```

and add an:

```text
Image
```

processing block followed by a:

```text
Classification
```

learning block.

---

## 6. Train

Reference training settings:

```text
Cycles:        10
Learning rate: 0.0005
```

Verify that the classifier performs correctly before benchmarking the hardware.

---

## 7. Export

Go to:

```text
Deployment
→ OpenMV Library
```

Build and download the deployment archive.

---

## 8. Copy the model

Copy:

```text
trained.tflite
labels.txt
```

to the OpenMV camera storage.

---

## 9. Open the benchmark script

Open:

```text
ei_classification_performance.py
```

in OpenMV IDE.

---

## 10. Run

Connect the camera and run the script.

Monitor:

```text
classification
inference time
FPS
```

in the OpenMV IDE console.

---

## 11. Change resolution

Repeat the experiment using different source-image resolutions.

Start with:

```text
240 × 240
```

and test larger frames where appropriate.

The original experiment reaches approximately:

```text
1024 × 1024
```

after switching the camera frame mode to SXGA.

---

## 12. Record measurements

Store results in:

```text
OpenMV RT1062 Edge Impulse.xlsx
```

For each configuration, record information such as:

```text
resolution
inference latency
FPS
idle current
inference current
```

---

# 📐 Benchmark methodology

For reproducible measurements, keep the following constant:

- same model
- same OpenMV firmware
- same lighting
- same target object
- same camera position
- same power supply
- same inference script
- same measurement equipment
- same model input size

Only modify one parameter at a time.

For example:

```text
Run 1 → 240 × 240
Run 2 → larger frame
Run 3 → larger frame
...
```

This makes the impact of camera resolution easier to isolate.

---

# 📁 Repository structure

```text
openMVClassificationPerformance/
├── LICENSE
├── OpenMV RT1062 Edge Impulse.xlsx
├── README.md
└── ei_classification_performance.py
```

### `ei_classification_performance.py`

MicroPython image-classification benchmark.

Used to evaluate:

- classifier output
- inference timing
- FPS
- different camera resolutions

### `OpenMV RT1062 Edge Impulse.xlsx`

Spreadsheet containing benchmark measurements collected during the experiment.

### `README.md`

Original repository documentation.

### `LICENSE`

MIT License.

---

# 🔬 Ideas for extending the project

1. **📊 Automated benchmark logging** — write latency, FPS, confidence, temperature and resolution directly to CSV or microSD after every inference.

2. **🔋 Measure energy per inference** — use a Joulescope or dedicated current monitor to compare models using joules rather than approximate current draw.

3. **🧠 Compare ML workloads** — benchmark image classification, FOMO object detection and multiple model sizes on the same RT1062 hardware.

---

# 📰 External references

## 🧠 Edge Impulse Expert Network

### OpenMV Cam RT1062 — Getting Started with Machine Learning

The complete project tutorial is published in the **Edge Impulse Expert Network**.

It documents:

- OpenMV RT1062 hardware
- 60-image dataset
- LEGO figure / blue ball classifier
- 96×96 impulse
- 10 training cycles
- `0.0005` learning rate
- OpenMV Library deployment
- MicroPython workflow
- inference-time measurements
- camera-resolution experiments
- power-consumption measurements
- standalone execution
- board status LEDs
- firmware recovery

**[OpenMV Cam RT1062 — Getting Started with Machine Learning](https://docs.edgeimpulse.com/projects/expert-network/getting-started-openmv-rt1062)**

---

## 📦 Edge Impulse Expert Projects repository

The tutorial is also maintained in the public Edge Impulse Expert Projects repository.

**[OpenMV RT1062 tutorial source — Edge Impulse GitHub](https://github.com/edgeimpulse/expert-projects/blob/main/readme/prototype-and-concept-projects/getting-started-openmv-rt1062.md)**

The project is included in the **Prototype and Concept Projects** section of the Edge Impulse Expert Network.

---

## ⚙️ Edge Impulse OpenMV deployment documentation

Current Edge Impulse documentation lists the **OpenMV Cam RT1062** as a supported deployment target.

It documents both:

```text
OpenMV Library
```

and:

```text
OpenMV Firmware
```

deployment methods.

**[Run OpenMV library or firmware — Edge Impulse](https://docs.edgeimpulse.com/hardware/deployments/run-openmv)**

---

# 📷 OpenMV references

## OpenMV Cam RT1062

Official product information:

**[OpenMV Cam RT1062](https://openmv.io/products/openmv-cam-rt)**

---

## OpenMV MicroPython documentation

**[OpenMV Cam RT1062 Quick Reference](https://docs.openmv.io/dev/openmvcam/quickref/openmv-rt1062.html)**

---

## OpenMV IDE

**[Download OpenMV IDE](https://openmv.io/pages/download)**

---

# 🎓 Academic reference

A 2025 University of Zaragoza engineering thesis discussing computer-vision platforms includes the **OpenMV Cam RT1062** and attributes its illustrated reference to **Roni Bandini, 2024**.

**[Diseño e Implementación de un Sistema HVAC Inteligente con Visión por Computadora — Universidad de Zaragoza](https://zaguan.unizar.es/record/164482/files/TAZ-TFG-2025-4661.pdf)**

---

# 📕 Contracultura Maker

More projects, technical experiments and context around embedded AI, Machine Learning, electronics and unconventional prototyping are collected in:

**[Contracultura Maker — book](https://bandini.medium.com/libro-de-contracultura-maker-94d1bb0d951c)**

---

# 📚 Useful references

- **[Edge Impulse](https://edgeimpulse.com/)**
- **[Edge Impulse Studio](https://studio.edgeimpulse.com/)**
- **[OpenMV Cam RT1062](https://openmv.io/products/openmv-cam-rt)**
- **[OpenMV documentation](https://docs.openmv.io/)**
- **[OpenMV IDE](https://openmv.io/pages/download)**
- **[Edge Impulse OpenMV deployment](https://docs.edgeimpulse.com/hardware/deployments/run-openmv)**
- **[OpenMV RT1062 Edge Impulse tutorial](https://docs.edgeimpulse.com/projects/expert-network/getting-started-openmv-rt1062)**

---

# 🔗 You may also be interested in...

Other projects by **Roni Bandini** involving Edge Impulse, embedded AI and computer vision.

## 👁️⚙️ Visual Anomaly

**Real-time visual anomaly detection on a Texas Instruments edge AI board using Edge Impulse FOMO-AD and a USB camera.**

It explores a more complex vision workload focused on defect detection and anomaly localization.

**[github.com/ronibandini/visualAnomaly](https://github.com/ronibandini/visualAnomaly)**

---

## 👁️🗂️ PunchedCards

**Image classification of punched-card-inspired binary patterns using Edge Impulse and a LattePanda IOTA.**

Another project centered on training, deploying and parsing a custom Edge Impulse image classifier.

**[github.com/ronibandini/PunchedCards](https://github.com/ronibandini/PunchedCards)**

---

## 🎙️⚡ Rubik Pi Audio Classification

**Real-time glass-break audio classification using Edge Impulse on the Thundercomm RUBIK Pi 3.**

It applies the same edge-inference workflow to audio instead of images and connects classification output to GPIO.

**[github.com/ronibandini/Rubik-Pi-AudioClassification](https://github.com/ronibandini/Rubik-Pi-AudioClassification)**

---

# 📜 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# 👤 Author

**Roni Bandini**

Maker, AI developer, electronic artist and writer.

- 🐙 GitHub: [@ronibandini](https://github.com/ronibandini)
- 💼 LinkedIn: [Roni Bandini](https://www.linkedin.com/in/ronibandini/)
- 📸 Instagram: [@ronibandini](https://www.instagram.com/ronibandini/)
- 🐦 X: [@RoniBandini](https://x.com/RoniBandini)
- ✍️ Medium: [bandini.medium.com](https://bandini.medium.com/)
- 🛠️ Hackster: [Roni Bandini](https://www.hackster.io/roni-bandini)
- 🔧 Hackaday.io: [Roni Bandini](https://hackaday.io/ronibandini)

Buenos Aires, Argentina.
