# MLops-Final-Project

[MLops-Final-Project](https://docs.google.com/document/d/1shFQ_QEM6-WWJpvHPlkx7_FjPbFmjwMKdknKFCaruxI/edit?tab=t.0)

## Intelligent Multimedia Processing (IMP) for Enterprises
Enterprises currently rely on manual searching through documents, audio, and video recordings, which is labor-intensive and inefficient. The IMP system automates the extraction and indexing of this multimedia data, allowing employees to directly query the information using natural language questions. Additionally, the system can automatically generate minutes of meetings from recorded meeting sessions, further increasing productivity and documentation accuracy. Key business metrics for evaluation include reduction in time spent searching for information, accuracy of retrieved answers, and quality of automatically generated meeting minutes.
<!-- 
Discuss: Value proposition: Your will propose a machine learning system that can be used in an existing business or service. (You should not propose a system in which a new business or service would be developed around the machine learning system.) Describe the value proposition for the machine learning system. What’s the (non-ML) status quo used in the business or service? What business metric are you going to be judged on? (Note that the “service” does not have to be for general users; you can propose a system for a science problem, for example.)
-->

### Contributors

<!-- Table of contributors and their roles. First row: define responsibilities that are shared by the team. Then each row after that is: name of contributor, their role, and in the third column you will link to their contributions. If your project involves multiple repos, you will link to their contributions in all repos here. -->

| Name                            | Responsible for                          | Link to their commits in this repo |
|---------------------------------|------------------------------------------|------------------------------------|
| All team members                |                                          |                                    |
| Akshat Mishra                   | data pipeline                            |                                    |
| Siddhant Mohan                  | model training                           |                                    |
| Mihir Khare                     | model serving & monitoring               |                                    |
| Nikita Gupta                    | continuous X pipeline                    |                                    |



### System diagram
![Editor _ Mermaid Chart-2025-04-03-012909](https://github.com/user-attachments/assets/2837f94c-4875-4c08-abb5-9f566b161687)

<!-- Overall digram of system. Doesn't need polish, does need to show all the pieces. Must include: all the hardware, all the containers/software platforms, all the models, all the data. -->

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. Name of data/model, conditions under which it was created (ideally with links/references), conditions under which it may be used. -->

|              | How it was created   | Conditions of use  |
|--------------|----------------------|--------------------|
| Data set 1   |os- AMI Meeting Corpus| LLM Fine Tuning    |
| Data set 2   |os-ICSI Meeting Corpus|  LLM Fine Tuning   |
| Base model 1 | Llama 7b             |                    |
| Base model 2 | VOSK                 |  (speech to text)  |
| Base model 3 | longformer-base-4096 |Text embedding model|


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), how much/when, justification. Include compute, floating IPs, persistent storage. The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan
Detailed Design Plan for Intelligent Multimedia Processing (IMP) System
Model Training and Training Platforms

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the diagram, (3) justification for your strategy, (4) relate back to lecture material, (5) include specific numbers. -->

#### Model training and training platforms
## Strategy

Our training strategy employs distributed training techniques for the Longformer-base-4096 embeddings model and fine-tuning of Llama 7B using LORA (Low-Rank Adaptation) to optimize performance while minimizing computational requirements. We'll implement a continuous training pipeline that automatically triggers retraining based on drift detection.
Relevant Parts of the Architecture

MLFlow for experiment tracking and model versioning
Ray clusters for distributed training
Longformer-base-4096, pretrained on long documents
Llama 7B with LORA optimizations
Model Registry for version control and deployment

Justification

Distributed Training: The ICSI Meeting Recorder Corpus (39GB) contains substantial audio data requiring efficient parallelized processing. Ray's distributed training framework enables us to scale across multiple nodes on Chameleon infrastructure.
Model Selection: Longformer-base-4096 (560MB) is specifically designed for long context (up to 4,096 tokens), making it ideal for meeting transcripts and lengthy documents while remaining small enough (560MB) for efficient deployment.
Parameter-Efficient Fine-tuning: LORA reduces the number of trainable parameters for Llama 7B by approximately 95%, enabling fine-tuning on limited resources while maintaining performance.

Relation to Lecture Material
This approach implements the distributed training patterns discussed in the "Training at Scale" lectures, specifically:

Data Parallelism through Ray's DDP implementation
Hyperparameter optimization via Ray Tune
Parameter-efficient fine-tuning techniques (LORA)
Experiment tracking best practices using MLFlow

## Specific Numbers

Training Infrastructure: 4 GPU nodes on Chameleon Cloud (each with NVIDIA T4)
Training Dataset: 39GB ICSI Meeting Corpus with approximately 75 hours of meeting recordings
Batch Size: 32 per GPU (128 global batch size)
Hyperparameter Tuning: 30 trials using Ray Tune with Bayesian optimization
Training Time: Approximately 12 hours for full embedding model training
LORA Configuration: Rank 8, alpha 16, reducing trainable parameters from ~7B to ~35M

## <!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, and which optional "difficulty" points you are attempting. -->

## Model serving and monitoring platforms
Strategy
We'll implement a multi-tier serving strategy with optimizations for both latency-sensitive and throughput-oriented workloads. FastAPI will serve as our API layer, with optimized inference paths for both CPU and GPU deployment options. Comprehensive monitoring will track model performance, drift, and system health.
Relevant Parts of the Architecture

FastAPI service layer
TorchServe for model deployment
Quantized models for CPU inference
GPU acceleration for batch processing
MLFlow dashboards for monitoring
Interactive data dashboard for visualization

Justification

Inference Optimization: Quantization reduces the Longformer model size by 75% (from 560MB to ~140MB) while maintaining 97%+ of accuracy, enabling faster CPU inference for low-latency requirements.
Multiple Serving Options: Different query patterns (interactive vs. batch processing) require different optimization strategies. Our architecture supports both GPU-accelerated batch processing and optimized CPU inference.
Comprehensive Monitoring: Early detection of model drift is crucial for maintaining accuracy in enterprise environments where data patterns evolve.

Relation to Lecture Material
This implements the serving concepts from "Model Serving" lectures:

Model optimization techniques (quantization, operator fusion)
Batch vs. real-time inference patterns
Zero-downtime deployment strategies
A/B testing for model deployments

Specific Numbers

Latency Target: <200ms for query processing (from request to embedding generation)
Throughput: 50 queries per second per node
Model Size Reduction: From 560MB to ~140MB through INT8 quantization
Monitoring Frequency: Performance metrics collected every 30 seconds
Alerting Threshold: Alert if accuracy drops below 92% or latency exceeds 500ms
Canary Deployment: 10% traffic to new model versions for validation
<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements,  and which optional "difficulty" points you are attempting. -->

#### Data pipeline
Strategy
Our data pipeline focuses on efficient processing of multimedia inputs, with separate paths for documents, audio, and video. We'll implement both batch processing for historical data and streaming capabilities for real-time meeting analysis. A comprehensive data dashboard will provide insights into data quality and processing efficiency.
Relevant Parts of the Architecture

Document, audio, and video processing pipelines
VOSK for speech-to-text conversion
Chunking module for text segmentation
Metadata extraction and storage
Chameleon persistent storage
Interactive data dashboard

Justification

Specialized Processing Paths: Different media types require specialized processing techniques. Our pipeline handles each appropriately while converging to a common text representation.
Persistent Storage Strategy: Enterprise data requires secure, persistent storage with proper versioning and access controls.
Streaming Processing: Real-time meeting analysis requires low-latency processing of audio streams, necessitating an optimized streaming pipeline.

Relation to Lecture Material
This implements the data engineering concepts from lectures:

ETL pipeline design
Data quality monitoring
Offline vs. online data processing
Feature store concepts (for embeddings and metadata)

Specific Numbers

Storage Allocation: 500GB persistent storage on Chameleon
Processing Throughput: 10MB/s for document processing, 1 hour of audio in <5 minutes
Chunking Configuration: Semantic chunks of 300-500 tokens with 50-token overlap
Data Retention: 90-day default retention with configurable policies
Dashboard Refresh: Real-time metrics with 15-second refresh interval
Error Budget: <0.5% processing failures with automatic retry capabilities
<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which optional "difficulty" points you are attempting. -->

#### Continuous X
Strategy
We'll implement a comprehensive CI/CD pipeline integrated with continuous training and monitoring to ensure both code and model quality. GitOps principles using ArgoCD will enable declarative infrastructure and application deployment, with PythonChi providing Infrastructure as Code capabilities.
Relevant Parts of the Architecture

GitHub repository for version control
CI/CD pipeline for testing and deployment
PythonChi for IaaC
Docker containerization
Kubernetes orchestration
ArgoCD for GitOps
Staged deployment patterns

Justification

GitOps Approach: Declarative infrastructure ensures consistency between environments and enables rapid rollback if needed.
Continuous Training: Automating the retraining process based on drift detection ensures model accuracy over time without manual intervention.
Staged Deployments: Progressive rollout of updates minimizes risk and enables validation before full deployment.

Relation to Lecture Material
This implements DevOps and MLOps concepts from lectures:

CI/CD pipeline design
Infrastructure as Code principles
Container orchestration patterns
GitOps deployment methodology
Continuous training workflows

Specific Numbers

Test Coverage: Minimum 85% code coverage for all components
Deployment Frequency: Support for multiple deployments per day if needed
Rollback Time: <5 minutes for reverting to previous stable version
Infrastructure Provisioning: <15 minutes from commit to complete environment setup
Deployment Stages: Development → Testing → Staging → Production with automated promotion
Model Validation: Automated evaluation against benchmark dataset (95% accuracy threshold) before promotion

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which optional "difficulty" points you are attempting. -->


