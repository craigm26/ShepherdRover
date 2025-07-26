# **ShepherdRover**

*AI-Powered Autonomous Field Rover for Precision Agriculture*

![ShepherdRover Banner](docs/images/shepherd_banner.png)

**ShepherdRover** is a **cutting-edge autonomous rover platform** that combines **multiple AI technologies** for agricultural scouting and decision-making. It leverages **Gemma 3n** across five innovative tracks to create the most advanced agricultural robotics system available.

## **🤖 AI Technology Stack**

ShepherdRover integrates **all five Kaggle AI technology tracks**:

### **🏆 LeRobot Prize Integration**
- **Gemma 3n Action Models** for autonomous decision-making
- **Real-time robotics control** using LeRobot framework
- **Dynamic path planning** based on field conditions
- **Intelligent obstacle avoidance** with contextual understanding

### **🚀 Jetson Prize Integration** 
- **NVIDIA Jetson Orin NX** for on-device AI processing
- **Real-time sensor fusion** and perception
- **Local inference** for critical decision-making
- **Edge computing** for low-latency responses

### **🦙 Ollama Prize Integration**
- **Local Gemma 3n deployment** via Ollama
- **Offline AI capabilities** for remote field operations
- **Privacy-preserving** data processing
- **Redundant AI systems** for reliability

### **⚡ Unsloth Prize Integration**
- **Fine-tuned Gemma 3n models** for agricultural tasks
- **Crop disease detection** and classification
- **Harvest readiness assessment** with high accuracy
- **Weather-adaptive** decision making

### **🌐 Google AI Edge Prize Integration**
- **Google AI Edge implementation** for efficient inference
- **Optimized model deployment** for resource-constrained environments
- **Battery-efficient** AI processing
- **Scalable edge computing** architecture

---

## **🎯 Core Capabilities**

**ShepherdRover** is designed to collect **multimodal field data** and provide **AI-powered insights**:

* **Autonomous Navigation** - ROS2-based with AI-enhanced path planning
* **Real-time Perception** - Multi-sensor fusion with on-device AI
* **Agricultural Intelligence** - Fine-tuned models for crop analysis
* **Edge Computing** - Local AI processing for immediate decisions
* **Fleet Coordination** - Multi-rover management (enterprise)

This repository contains the **open components** of ShepherdRover:

* ROS2-based navigation stack with LeRobot integration
* Jetson-optimized firmware and sensor modules
* Ollama deployment configurations
* Unsloth fine-tuning pipelines
* Google AI Edge implementation
* Mechanical & electrical designs (CAD, wiring diagrams)
* Bill of Materials (BOM) & assembly instructions

> **Note:** Advanced AI models, proprietary fine-tuned weights, and enterprise fleet management tools are **proprietary** and available under separate licensing.

---

## **Licensing Map**

Below is a visual overview of which components are **open-source** (green) vs **proprietary** (orange):

```
                        ┌─────────────────────────────┐
                        │       Farmhand AI           │
                        │  Proprietary (Gemma 3n AI)  │
                        │  - Fine-tuned Models        │
                        │  - Advanced Reasoning       │
                        │  - Fleet Management Tools   │
                        └─────────────┬──────────────┘
                                      │
                                      │ API (Proprietary)
                                      │
         ┌────────────────────────────▼─────────────────────────────┐
         │                     ShepherdRover                        │
         │    (This Repository – Open-Source Components)            │
         │─────────────────────────────────────────────────────────│
         │   Open:                                                │
         │    - ROS2 + LeRobot Integration                        │
         │    - Jetson Firmware & Optimizations                   │
         │    - Ollama Deployment Configs                         │
         │    - Unsloth Training Pipelines                        │
         │    - Google AI Edge Implementation                     │
         │    - Mechanical & Electrical Designs                   │
         │    - BOM & Assembly Documentation                      │
         │                                                         │
         │   Proprietary Add-ons (Separate License):               │
         │    - Fine-tuned Model Weights                          │
         │    - Farmhand AI Connector Modules                     │
         │    - Premium Sensor Fusion Models                      │
         └─────────────────────────────────────────────────────────┘
```

* **Green = Open-Source (Apache 2.0)** → Free for personal and commercial use (with attribution).
* **Orange = Proprietary** → Available under a separate commercial license.

---

## **Repository Structure**

```
shepherd-rover/
├── bom/                # Bill of Materials (JSON + PDF)
├── cad/                # Mechanical designs (CAD)
├── docs/               # Assembly & wiring guides
├── firmware/           # Jetson-optimized firmware
├── navigation/         # ROS2 + LeRobot navigation stack
├── perception/         # Multi-sensor fusion modules
├── ai/                 # AI technology implementations
│   ├── lerobot/        # LeRobot action models
│   ├── jetson/         # Jetson optimizations
│   ├── ollama/         # Ollama deployment configs
│   ├── unsloth/        # Fine-tuning pipelines
│   └── edge/           # Google AI Edge implementation
├── api/                # API definitions for Farmhand AI
├── CONTRIBUTING.md     # Contribution guidelines
├── CLA.md              # Contributor License Agreement
└── LICENSE             # Apache 2.0 License (with hybrid notes)
```

---

## **Getting Started**

### **1. Clone the Repository**

```bash
git clone https://github.com/YourOrg/shepherd-rover.git
cd shepherd-rover
```

### **2. Review the BOM**

Check `bom/` for a full list of required components including:
- **NVIDIA Jetson Orin NX** development kit
- **Multi-spectral cameras** and sensors
- **Robust chassis** and drive system
- **AI-optimized** computing platform

Estimated cost: **~$3,500 USD** (including Jetson hardware)

### **3. Setup AI Technologies**

```bash
# Install Ollama and Gemma 3n
cd ai/ollama
./setup_ollama.sh

# Setup Unsloth fine-tuning environment
cd ../unsloth
pip install -r requirements.txt

# Configure Google AI Edge
cd ../edge
./setup_edge.sh

# Setup LeRobot framework
cd ../lerobot
pip install -r requirements.txt
```

### **4. Build the Rover**

Follow the step-by-step assembly instructions in `docs/`.

### **5. Run the AI-Enhanced Navigation Stack**

```bash
cd navigation
colcon build
source install/setup.bash

# Launch with all AI technologies
ros2 launch shepherd_navigation ai_bringup.launch.py
```

---

## **AI Technology Details**

### **LeRobot Integration**
- **Action Models**: Gemma 3n fine-tuned for agricultural robotics
- **Real-time Control**: Sub-second decision making for field navigation
- **Context Awareness**: Understanding of crop types, soil conditions, weather

### **Jetson Optimization**
- **TensorRT Acceleration**: Optimized inference for real-time processing
- **Multi-modal Fusion**: Camera, LiDAR, and sensor data integration
- **Power Management**: Efficient AI processing for extended field operations

### **Ollama Local Deployment**
- **Offline Capability**: AI reasoning without internet connectivity
- **Privacy First**: All data processed locally on the rover
- **Redundant Systems**: Backup AI for critical decision making

### **Unsloth Fine-tuning**
- **Agricultural Models**: Specialized for crop disease detection
- **Weather Adaptation**: Models that learn from local conditions
- **Continuous Learning**: Models that improve with field experience

### **Google AI Edge**
- **Efficient Inference**: Optimized for resource-constrained environments
- **Battery Optimization**: AI processing that maximizes field time
- **Scalable Architecture**: Easy deployment across rover fleet

---

## **Contributing**

We welcome contributions to the **open components** of ShepherdRover!

1. Read the [CONTRIBUTING.md](./CONTRIBUTING.md).
2. Sign the [Contributor License Agreement (CLA)](./CLA.md).
3. Submit your changes via a Pull Request using the provided template.

**AI Contributions**: We especially welcome contributions to the AI technology implementations in the `ai/` directory.

---

## **Licensing**

* **Open-Source Components:** Licensed under [Apache 2.0](./LICENSE).
* **Proprietary Modules:** Fine-tuned model weights, Farmhand AI connectors, and enterprise features are **not covered** by this license and require a commercial agreement.

---

## **Roadmap**

* **Phase 1:** Open-source release of AI-enhanced BOM, ROS2+LeRobot stack, and build guides (Hackathon prototype).
* **Phase 2:** DIY kit program with Jetson and AI capabilities for hobbyists and universities.
* **Phase 3:** Pre-built Shepherd units + Farmhand AI enterprise integration with all five AI technologies.

---

## **Contact**

* **Website:** [farmhandai.com](https://farmhandai.com)
* **Issues & Support:** [GitHub Issues](../../issues)
* **Email:** [support@farmhandai.com](mailto:support@farmhandai.com)
* **AI Research:** [ai@farmhandai.com](mailto:ai@farmhandai.com)

---

