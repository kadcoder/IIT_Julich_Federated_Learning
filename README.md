# IIT_Julich_Federated_Learning
### OA : The ML model is trained individually with all silos data

### OB: The ML model is trained with all silo data in a centralized setup

### Pre‑H (Global) (Id1/exp 2) : We will train the global model with the global data, train the local model with local silo data and updating the global model weights with the gradients.

### Post‑H‑m1 (Global) (Id1/exp 4): First we will train the global model with the global data, then local silo data and global data feature harmonization using neuroCombat followed by training the local silos with harmonized data and updating the global model weights with the gradients.

### Post‑H‑m2 (Global) (Id2/exp 1): We will train the global model with the global data, train the local model with local silo data followed by harmonizing the local gradients and updating the global model weights with the aggregated (harmonized local) gradients.

### Post‑H‑m3 (Global) (Id2/exp 3): We will train the global model with the global data, train the local model with local silo data, fine-tuned by global data, then followed by harmonizing the local gradients (gradients extracted from local ML model) and global (gradients extracted from locally fine-tuned ML model) gradients and updating the global model weights with the aggregated (harmonized) gradients.

### Post‑H‑m4 (Global) (Id2/exp 2): We will train the global model with the global data, train the local model with local silo data, fine-tuned by global data, then followed by harmonizing the global (gradients extracted from locally fine-tuned ML model) gradients and updating the global model weights with the aggregated (harmonized) gradients.
