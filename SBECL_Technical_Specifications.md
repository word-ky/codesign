**Core Technical Specifications: Secure Bi-directional Edge-Cloud Linkage Protocol (SBECL)**
**Module:** Heterogeneous Network Communication Layer | **Version:** v2.1.0 | **Status:** Deployed

### 1. Design Rationale

In distributed edge computing scenarios, we confront complex challenges associated with heterogeneous networks. Specifically, to address the connection instability caused by the dynamic IP allocation of commercial cloud servers (e.g., AutoDL) and the NAT traversal difficulties faced by edge devices (e.g., NVIDIA Jetson TX2) within intranet environments, traditional direct connection schemes fail to meet high-availability requirements.

To this end, we have designed and implemented a custom communication protocol based on encrypted tunneling technology—**SBECL (Secure Bi-directional Edge-Cloud Linkage)**. This protocol aims to construct a robust transmission substrate within heterogeneous network environments, ensuring high-reliability and low-latency data interaction between the Edge Client and the Cloud Server.

### 2. Protocol Architecture

The SBECL protocol eschews traditional polling mechanisms in favor of an **Event-Driven** transmission architecture. We initialize an encrypted persistent connection at the Transport Layer and achieve deep decoupling of logical links through a custom port mapping mechanism.

#### 2.1. Phase I: Initialization & Cryptographic Handshake

The primary stage of protocol startup involves establishing a secure control plane. The edge node actively initiates a connection request to the cloud, executing a **Cryptographic Handshake** process based on RSA-4096 key pairs.

* **Authentication:** Verifies the device fingerprint of the edge node to prevent unauthorized device access.
* **Tunneling:** Upon a successful handshake, SBECL constructs an encapsulation layer atop the TCP/IP protocol stack to traverse complex NAT gateways and firewall rules.

#### 2.2. Phase II: Logical Decoupling & Virtualization

To mitigate resource contention between control flows and data flows (Head-of-Line Blocking), the SBECL protocol establishes bi-directionally isolated virtual channels via port mapping mechanisms, thereby achieving **Full-Duplex Communication**:

* **Channel A: Uplink Command/Inference Channel (Port 6006)**
* *Definition:* Edge Node → Cloud Server
* *Function:* Dedicated to transmitting high-frequency inference requests, device telemetry data, and heartbeat keep-alive packets.


* **Channel B: Downlink Data/Control Channel (Port 6008)**
* *Definition:* Cloud Server → Edge Node
* *Function:* Dedicated to distributing large-capacity model weight updates, **IoVT** aggregation parameters, and system control commands.



Through this logical link decoupling, downlink control commands can still arrive with priority via an independent channel even during uplink congestion, ensuring system controllability.

### 3. Implementation Specifications

#### 3.1. Protocol Bootstrap

The following commands are used to construct the SBECL tunnel at the transport layer and complete virtual port mapping and compression algorithm negotiation during the initialization phase:

```bash
# SBECL Transport Layer Initialization
# - Encrypted Tunnel Establishment
# - Logical Link Decoupling (6006 Uplink / 6008 Downlink)
ssh -CNg \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -L 6006:127.0.0.1:6006 \
    -R 6008:127.0.0.1:6008 \
    root@connect.westb.seetacloud.com -p 23183

```

#### 3.2. Service Mounting

After tunnel establishment, application-layer services achieve transparent transmission by mounting to the local virtual interface (Localhost Interface):

**Edge Side (Client Worker):**

```bash
# Connects to the Uplink Channel via Virtual Port
python client_worker.py --server_ip 127.0.0.1 --port 6006

```

**Cloud Side (Server Main):**

```bash
# Listens on the Virtual Uplink Port for Handshake Requests
python server_main.py --port 6006

```

### 4. Technical Characteristics

This protocol scheme possesses three core advantages, which not only resolve fundamental connectivity issues but also enhance overall system robustness:

1. **Robust NAT Traversal:**
Utilizing a reverse connection mechanism, the SBECL protocol seamlessly penetrates multi-level NAT gateways. Whether the edge device is located in a 4G/5G mobile network or a strict enterprise intranet, it remains addressable by the cloud without requiring a public IP.
2. **Bi-directional Isolated Communication:**
By constructing independent uplink and downlink virtual channels, the protocol achieves traffic isolation. This not only optimizes bandwidth utilization but also architecturally eliminates "communication deadlock" phenomena caused by unidirectional link failures.
3. **Enterprise-Grade Security:**
All transmitted data is encapsulated via the SSHv2 protocol, employing the AES-GCM algorithm for **End-to-End Encryption**. This mechanism ensures that sensitive inference data and model parameters are protected from eavesdropping or tampering during public network transmission, fully satisfying data privacy compliance requirements in **IoVT** scenarios.

