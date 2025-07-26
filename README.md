# **ShepherdRover**

*Autonomous Field Rover for Precision Agriculture*

![ShepherdRover Banner](docs/images/shepherd_banner.png)

**ShepherdRover** is an **open-source autonomous rover platform** for agricultural scouting. It's designed to collect **multimodal field data** — kernel samples, multispectral imagery, and positional metrics — and integrate seamlessly with **Farmhand AI** for **harvest readiness insights**.

This repository contains the **open components** of ShepherdRover:

* ROS2-based navigation stack
* Firmware for sensor modules
* Mechanical & electrical designs (CAD, wiring diagrams)
* Bill of Materials (BOM) & assembly instructions

> **Note:** Farmhand AI integration, advanced perception models, and enterprise fleet management tools are **proprietary** and available under separate licensing.

---

## **Licensing Map**

Below is a visual overview of which components are **open-source** (green) vs **proprietary** (orange):

```
                        ┌─────────────────────────────┐
                        │       Farmhand AI           │
                        │  Proprietary (Gemma 3n AI)  │
                        │  - Conversational Assistant │
                        │  - Advanced Models          │
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
         │    - ROS2 Navigation Stack                              │
         │    - Base Sensor Firmware                               │
         │    - Mechanical & Electrical Designs (CAD, Diagrams)    │
         │    - BOM & Assembly Documentation                       │
         │                                                         │
         │   Proprietary Add-ons (Separate License):               │
         │    - Farmhand AI Connector Modules                      │
         │    - Premium Sensor Fusion Models                       │
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
├── firmware/           # Microcontroller firmware
├── navigation/         # ROS2 navigation stack
├── perception/         # Sensor integration modules
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

Check `bom/` for a full list of required components and estimated costs.
A [PDF build guide](bom/Shepherd_BOM.pdf) is also included.

### **3. Build the Rover**

Follow the step-by-step assembly instructions in `docs/`.

### **4. Run the ROS2 Stack**

```bash
cd navigation
colcon build
source install/setup.bash
ros2 launch shepherd_navigation bringup.launch.py
```

---

## **Contributing**

We welcome contributions to the **open components** of ShepherdRover!

1. Read the [CONTRIBUTING.md](./CONTRIBUTING.md).
2. Sign the [Contributor License Agreement (CLA)](./CLA.md).
3. Submit your changes via a Pull Request using the provided template.

---

## **Licensing**

* **Open-Source Components:** Licensed under [Apache 2.0](./LICENSE).
* **Proprietary Modules:** Farmhand AI connectors, Gemma 3n-based perception models, and enterprise features are **not covered** by this license and require a commercial agreement.

---

## **Roadmap**

* **Phase 1:** Open-source release of BOM, ROS2 stack, and build guides (Hackathon prototype).
* **Phase 2:** DIY kit program for hobbyists and universities.
* **Phase 3:** Pre-built Shepherd units + Farmhand AI enterprise integration.

---

## **Contact**

* **Website:** [farmhandai.com](https://farmhandai.com)
* **Issues & Support:** [GitHub Issues](../../issues)
* **Email:** [support@farmhandai.com](mailto:support@farmhandai.com)

---

