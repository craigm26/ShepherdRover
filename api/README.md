# ShepherdRover API

This directory contains API definitions and interfaces for integrating ShepherdRover with Farmhand AI.

## Overview

The API layer provides standardized interfaces for:

- **Data Collection** - Field sensor data and imagery
- **Command & Control** - Remote operation and mission planning
- **Fleet Management** - Multi-rover coordination (enterprise feature)
- **Analytics Integration** - Data processing and insights

## API Structure

```
api/
├── openapi/              # OpenAPI 3.0 specifications
├── examples/             # API usage examples
├── schemas/              # Data schemas and models
├── client/               # Client libraries and SDKs
└── docs/                 # API documentation
```

## Open-Source Components

### Data Collection API
- **Field Data Upload** - Sensor readings, GPS tracks, imagery
- **Mission Status** - Rover position, battery, health
- **Error Reporting** - Diagnostics and troubleshooting

### Basic Control API
- **Mission Planning** - Route definition and waypoints
- **Emergency Stop** - Safety controls
- **Status Monitoring** - Real-time rover state

## Proprietary Components

> **Note:** The following APIs require a commercial license:

- **Farmhand AI Integration** - Advanced AI reasoning and insights
- **Fleet Management** - Multi-rover coordination and optimization
- **Advanced Analytics** - Predictive modeling and recommendations
- **Enterprise Features** - User management, reporting, compliance

## Authentication

- **Open APIs:** API key authentication
- **Proprietary APIs:** OAuth 2.0 with enterprise credentials

## Rate Limits

- **Open APIs:** 1000 requests/hour
- **Proprietary APIs:** Based on subscription tier

## Getting Started

1. Review the OpenAPI specifications in `openapi/`
2. Check examples in `examples/`
3. Use client libraries in `client/`
4. Contact support for proprietary API access

## Support

- **Open APIs:** GitHub Issues
- **Proprietary APIs:** [support@farmhandai.com](mailto:support@farmhandai.com) 