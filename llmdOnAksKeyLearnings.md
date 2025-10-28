# **LLM-D on AKS – Problems, Solutions & Key Learnings**

Successfully deployed **llm-d** "Intelligent Inference Scheduling" well-lit path on AKS. Validated end-to-end workflow from cluster setup to inference testing using H100 and A10 NodePools.

- **Problem:** Decode pods fail on GPU due to high memory utilization  
  **Solution:** Reduce `--gpu-memory-utilization` (e.g., from 0.9 → 0.85)  
  **Learning:** Always tune GPU memory settings per workload and GPU type.

- **Problem:** Istio gateway returns 404 for `/v1/health`  
  **Solution:** Apply HTTPRoute CRD and configure LoadBalancer health probe annotations:  
    ```yaml
    service.beta.kubernetes.io/port_80_health-probe_interval: "59"
    service.beta.kubernetes.io/port_80_health-probe_request-path: "/health"
    ```  
  **Learning:** Correct health probe path and interval prevent 404s and excessive log spam. Deployment order and routing configuration are critical.

- **Problem:** Hugging Face API access fails  
  **Solution:** Create Kubernetes secret with HF token in the correct namespace  
  **Learning:** Proper secret management is essential for model access.

- **Problem:** GPU metrics not visible  
  **Solution:** Deploy Prometheus + Grafana + DCGM Exporter for access  
  **Learning:** Monitoring stack is necessary for GPU utilization, inference latency, and memory tracking.

- **Problem:** Non-GPU workloads scheduled on GPU nodes  
  **Solution:** Apply taints on GPU nodes when creating NodePools and matching tolerations on deployements.
  **Learning:** Node taints/tolerations ensure correct workload placement and efficient resource usage.
