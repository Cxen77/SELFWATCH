# Related Work Comparison Matrix

> Systematic comparison of SELFWATCH against existing MOT methods.

---

## Tracker Comparison

| Feature | DeepSORT | ByteTrack | BoT-SORT | StrongSORT++ | OC-SORT | SELFWATCH |
|---------|----------|-----------|----------|-------------|---------|-----------|
| Appearance Matching | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ |
| Motion Model | Kalman | Kalman | Kalman+CMC | Kalman+ECC | OOS Kalman | Kalman |
| Persistent Memory | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Occlusion Reasoning | ❌ | Low/High | Camera motion | ❌ | OOS | Phantom |
| Trajectory Ownership | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-Signal Fusion | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (6-weight) |
| Identity Memory | ❌ | ❌ | ❌ | ❌ | ❌ | Active+Warm+Archive |
| Deferred Decisions | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Gait Signal | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Scene Topology | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Contradiction Detection | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

## MOT17 Results Comparison (to be filled)

| Method | MOTA | IDF1 | HOTA | ID Sw. | FPS |
|--------|------|------|------|--------|-----|
| ByteTrack | 80.3 | 77.3 | 63.1 | 2196 | 30 |
| BoT-SORT | 80.5 | 80.2 | 65.0 | 1212 | 9 |
| StrongSORT++ | 79.6 | 79.5 | 64.4 | 1194 | 7 |
| OC-SORT | 78.0 | 77.5 | 63.2 | 1950 | 28 |
| **SELFWATCH** | **—** | **—** | **—** | **—** | **18.6** |
