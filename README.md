 ./# HealthQueue - Smart Healthcare Management System

## 🚀 Technical Overview
HealthQueue is a modern healthcare management system built with cutting-edge technologies to provide intelligent health assistance and facility management.

## 🛠️ Tech Stack

### Frontend Architecture
- **Framework**: Next.js 14 with App Router
- **UI Components**: 
  - Shadcn/ui for consistent design
  - Custom React components
  - TailwindCSS for styling
  - Radix UI primitives
- **Maps Integration**: 
  - Leaflet.js for interactive facility mapping
  - OpenStreetMap for geographical data
- **State Management**:
  - React Context API
  - Custom hooks for state logic

### Backend Services
- **Database & Authentication**: 
  - Supabase
    - PostgreSQL database
    - Real-time subscriptions
    - Row Level Security (RLS)
    - Auth with email/password
    - Secure JWT handling
  
- **AI Integration**:
  - Google Gemini Pro
    - Restricted medical knowledge base
    - Safety-first response generation
    - Context-aware health guidance
    - Symptom analysis with medical guidelines

### 1. Smart Healthcare Access
- **AI-Powered Health Assistant (Dr. DiagnosAI)**
  - Intelligent symptom analysis and preliminary health guidance
  - 24/7 accessible health information and support
  - Personalized health recommendations
  - Emergency situation recognition and guidance

### 2. Smart Facility Management
- **Real-time Healthcare Facility Mapping**
  - Interactive map showing nearby healthcare facilities
  - Real-time availability updates
  - Smart routing to nearest facilities
  - Facility type filtering (hospitals, clinics, pharmacies)

### 3. Digital Health Records
- **Secure Patient Profiles**
  - Centralized health information storage
  - Medical history tracking
  - Medication management
  - Allergy and condition tracking
  - Secure data encryption

### 4. Smart Healthcare Resource Optimization
- **Usage Analytics and Load Balancing**
  - Intelligent patient distribution
  - Resource utilization tracking
  - Peak hour management
  - Service optimization recommendations

### 5. Smart Emergency Response
- **Emergency Support System**
  - Quick access to emergency services
  - Real-time emergency guidance
  - First aid instructions
  - Emergency facility location services

## 🔑 Key Smart City Integration Points

1. **Data-Driven Healthcare**
   - AI-powered health analysis
   - Predictive healthcare trends
   - Population health monitoring
   - Resource allocation optimization

2. **Digital Infrastructure**
   - Cloud-based healthcare services
   - Secure data transmission
   - Real-time updates and notifications
   - Cross-platform accessibility

3. **Smart City Connectivity**
   - Integration with city health systems
   - Connected healthcare network
   - Digital health records sharing
   - Emergency response coordination

## 💻 Technical Implementation

### Core Features & Implementation

#### 1. AI Health Assistant (DiagnosAI)
```typescript
// Implemented using Gemini Pro with:
- Medical knowledge restriction
- Symptom analysis algorithms
- Emergency situation detection
- Safe response generation
- Context preservation
```

#### 2. Facility Management System
```typescript
// Built with:
- Real-time Supabase subscriptions
- Geospatial queries
- Dynamic facility status updates
- Distance-based sorting
```

#### 3. User Profiles & Health Records
```typescript
// Implemented using:
- Supabase secure storage
- Encrypted health data
- Real-time updates
- Row Level Security
```

### API Architecture
- **REST Endpoints**:
  - `/api/chat`: AI health assistant interface
  - `/api/report`: Health report generation
  - `/api/places`: Facility management
  - `/api/profile`: User profile management
  - `/api/doctor-thread`: Doctor communication

### Security Implementation
- Location-based Services
- AI Health Analysis
- Real-time Monitoring
- Secure Data Handling
- Emergency Response System

## 🔐 Security & Privacy

- End-to-end encryption
- HIPAA-compliant data handling
- Secure authentication
- Privacy-first design
- Data protection measures

## 🌱 Environmental Impact

- Paperless health records
- Optimized resource utilization
- Reduced unnecessary hospital visits
- Smart routing for reduced emissions
- Digital-first approach

## 🎯 Smart City Benefits

1. **Healthcare Accessibility**
   - 24/7 health support
   - Remote health guidance
   - Equal access to healthcare
   - Reduced wait times

2. **Resource Optimization**
   - Better resource allocation
   - Reduced healthcare costs
   - Improved facility utilization
   - Efficient patient distribution

3. **Emergency Preparedness**
   - Quick emergency response
   - Better crisis management
   - Real-time emergency guidance
   - Coordinated emergency services

4. **Public Health Management**
   - Population health monitoring
   - Disease outbreak tracking
   - Health trend analysis
   - Preventive healthcare measures

## 📱 Smart Features for Citizens

- **Easy Access**: Simple, intuitive interface for all ages
- **Real-time Updates**: Instant access to healthcare information
- **Smart Navigation**: Efficient routing to healthcare facilities
- **Digital Health Management**: Personal health tracking and management
- **Emergency Support**: Quick access to emergency services and guidance

## 🔄 Future Smart City Integration

- Integration with smart city transportation systems
- Connected ambulance services
- Smart hospital management systems
- IoT health monitoring devices
- City-wide health data analytics

## 🚀 Getting Started

1. Clone the repository
\`\`\`bash
git clone https://github.com/Impact-10/HealthQueue.git
\`\`\`

2. Install dependencies
\`\`\`bash
pnpm install
\`\`\`

3. Set up environment variables
\`\`\`bash
cp .env.example .env.local
\`\`\`

4. Start the development server
\`\`\`bash
pnpm dev
\`\`\`

## 🧠 Medical NLP Backend (FastAPI)

The `backend/` folder hosts a FastAPI service that wraps four lightweight, open-source medical NLP models. Each model runs in **inference** mode by default and exposes a dedicated REST endpoint for question answering or entity extraction.

### 1. Create & activate a Python environment

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer not to manage a virtual environment, install the essential dependencies directly:

```bash
pip install fastapi uvicorn transformers torch
```

### 2. Configure model modes and handles

Set the following environment variables to enforce live inference. They already default to `inference`, but declaring them explicitly helps during demonstrations or reviews.

```bash
export MEDALPACA_MODE=inference
export BIOGPT_MODE=inference
export LONGFORMER_MODE=inference
export PUBMEDBERT_MODE=inference

# Optional overrides
# Lightweight checkpoints only — the backend now rejects any override that references
# 13B+ (or otherwise unapproved) artifacts. The values below are the only supported
# repositories unless you extend `ALLOWED_MODEL_REPOS` in `backend/config.py`.
export MEDALPACA_MODEL_NAME="medalpaca/medalpaca-7b"
export BIOGPT_MODEL_NAME="microsoft/biogpt"
export LONGFORMER_MODEL_NAME="yikuan8/Clinical-Longformer"
export PUBMEDBERT_MODEL_NAME="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Optional shared cache folder to keep downloads out of your default Hugging Face dir
export MODEL_CACHE_DIR="/absolute/path/to/hf-cache"
```

The first request downloads the model weights from Hugging Face. Keep roughly 12 GB of free disk space and at least 16 GB of RAM available. On constrained CPUs you can set `*_LOCAL_ONLY=1` after the initial download or point `*_MODEL_NAME` to a quantised variant.

**Guardrails**: the FastAPI wrappers will raise `HTTP 503` if you attempt to set
`*_MODE=inference` but the required checkpoint is missing or oversized. Any
`*_MODEL_NAME` overriding the allowlisted repositories above will fail fast with a
clear error message so that 13B+ downloads never start.

### 3. Run the FastAPI server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Once the server boots it logs the resolved `MODEL_CACHE_DIR` (if provided) so you
can confirm the cache policy before triggering any downloads.

### 4. Smoke-test the endpoints

```bash
curl -X POST http://localhost:8000/api/medalpaca \
  -H "Content-Type: application/json" \
  -d '{"question": "What are early signs of diabetes?"}'

curl -X POST http://localhost:8000/api/biogpt \
  -H "Content-Type: application/json" \
  -d '{"question": "Suggest a differential diagnosis for chronic cough."}'

curl -X POST http://localhost:8000/api/clinical_longformer \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize key findings", "context": "Patient reports chest pain and dizziness..."}'

curl -X POST http://localhost:8000/api/pubmedbert \
  -H "Content-Type: application/json" \
  -d '{"text": "The patient was started on metformin 500 mg for type 2 diabetes."}'
```

Each payload keeps the request small enough that even CPU-only inference completes
quickly. All four calls should return `mode: "inference"` in the response metadata;
if a model fails to hydrate you will instead see `HTTP 503` with the exact guardrail
message (for example, if the cache is missing or you pointed to an unsupported repo).

Each response echoes the question/text, returns an `answer`, and includes a model badge (name, key, mode) plus the full `structuredResponse` envelope. When inference is required but weights are missing, the API returns **HTTP 503** with the failure reason instead of silently falling back.

### REST endpoint summary

| Endpoint | Purpose | Payload | Notes |
| --- | --- | --- | --- |
| `POST /api/biogpt` | Medical Q/A (DistilGPT-2) | `{ question, max_length?, temperature? }` | Fast, lightweight general-purpose language model for medical Q&A. |

### Troubleshooting

- **RuntimeError / HTTP 503** – Inference requested but weights unavailable. Check disk, Hugging Face access, or adjust `*_MODEL_NAME`.
- **CUDA OOM** – Run on CPU (`CUDA_VISIBLE_DEVICES=""`) or select a quantised checkpoint.
- **Slow responses** – Reduce `max_length`, switch to a distilled model, or increase hardware resources.
- **Download stalls** – Enable `HF_HUB_ENABLE_HF_TRANSFER=1` for resumable downloads or prefetch weights.

## 🧩 Frontend integration & comparison view


The chat interface (`components/chat/chat-interface.tsx`) now:
- Only provides a model selector for DistilGPT-2 (lightweight mode).
- Routes each request through `app/api/chat/route.ts` to the `/api/biogpt` FastAPI endpoint.
- Displays an inference badge per assistant response.
## ⚡ Model Setup Instructions

This project now uses the small, fast [DistilGPT-2](https://huggingface.co/distilgpt2) model for all medical Q&A endpoints. No large downloads or special hardware required.

**To pre-download the model weights:**

```
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('distilgpt2'); AutoTokenizer.from_pretrained('distilgpt2')"
```

Or simply start the backend and the model will be downloaded automatically on first use.

**No additional configuration is needed.**

Set `DIAGNOSAI_BACKEND_URL` in `.env.local` if the FastAPI server is hosted elsewhere. To compare all models for a single prompt, pick **“Ensemble (All Models)”** in the dropdown; the UI highlights which models produced genuine inference results.

## 📚 Recommended workflow

1. Launch the FastAPI backend in inference mode.
2. Start the Next.js frontend with `pnpm dev`.
3. Ask health-related questions via the chat UI or call the REST endpoints directly.
4. Capture responses and metadata (model badge + mode) for demonstrations or grading reviews.

## 🧪 MedAlpaca-style fine-tuning (synthetic diagnosis Q/A)

This repo includes a lightweight, no-external-API pipeline to generate a synthetic dataset of 1,000+ diagnosis Q/A pairs and fine-tune a causal LM with Hugging Face Transformers Trainer. By default it uses `distilgpt2` to run on CPU; you can point to a MedAlpaca checkpoint on a GPU box.

### 1) Generate dataset (JSON/JSONL/CSV)

```bash
python backend/scripts/generate_qna_dataset.py
# writes backend/data/med_qna.{jsonl,json,csv}
```

Each record:

```json
{"question": "Describe your medical question here.", "answer": "Provide a correct, clear, and concise diagnosis or guidance here."}
```

### 2) Fine-tune with Transformers Trainer (causal LM)

CPU-friendly default:

```bash
# Optional knobs
export BASE_MODEL=distilgpt2
export EPOCHS=1 BATCH_SIZE=2 EVAL_BATCH_SIZE=2 MAX_LENGTH=256 LR=5e-5

python backend/scripts/train_medalpaca.py
# saves to ./medalpaca-custom-diagnosis
```

Train real MedAlpaca (requires strong GPU, VRAM > 14GB typically):

```bash
export BASE_MODEL=medalpaca/medalpaca-7b
python backend/scripts/train_medalpaca.py
```

### 3) Test the fine-tuned model

```bash
python backend/scripts/test_medalpaca.py
```

This prints answers for representative questions to confirm improved, context-aware responses.

## 👥 Contributors
- [Your Team Members]

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links
- [Project Demo](https://your-demo-link.com)
- [Documentation](https://your-docs-link.com)
- [API Reference](https://your-api-docs.com)

---
*This project was developed as part of the Smart City Design course, implementing smart healthcare solutions for modern urban environments.*