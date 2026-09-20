# 🤖 AI Customer Support Lab

### CrewAI Multi-Agent Buildathon · Guardrails · Evaluation · Controlled Agent Experiments

A production-style customer-support application built to learn not only **how to use AI agents**, but also **when agent-based orchestration actually adds value**.

> **Build → Experiment → Learn → Harden → Deploy**

---

## 🚀 Live Application

### [Open the Live VPS Application](https://crewai-support.srv1965124.hstgr.cloud)

**Deployment:** Docker · Hostinger VPS · Traefik · Let's Encrypt HTTPS

**GitHub Repository:**  
https://github.com/arun-srinivasan-builds/CrewAI-Customer-Support

---

# 📌 Project Overview

This project started as a **CrewAI Buildathon** to implement a sequential three-agent customer-support workflow.

Rather than stopping after the agents worked, the application was extended into a controlled learning lab to understand:

- How multi-agent orchestration behaves in a fixed workflow
- When explicit Python orchestration may be simpler
- Where runtime agent decisions become useful
- How guardrails should surround an AI workflow
- How generated answers and tool decisions should be evaluated
- How session memory supports follow-up conversations
- How unnecessary model and search calls can be reduced
- How to move an AI application from local development to a live HTTPS deployment

> **Core Engineering Principle:** Use agents deliberately, not by default. Keep deterministic work deterministic and introduce runtime reasoning where it creates meaningful value.

---

# 🧭 Project Journey

| Stage | Implementation |
|---|---|
| **Build** | Three-agent CrewAI customer-support workflow |
| **Experiment** | Python vs CrewAI and fixed routing vs runtime capability selection |
| **Learn** | Understand where agents provide meaningful value |
| **Harden** | Guardrails, temporal validation, evals, memory, caching and trace validation |
| **Deploy** | Docker → Hostinger VPS → Traefik → Let's Encrypt HTTPS |

---

# 🤖 Buildathon Core — Three-Agent Workflow

The Buildathon implementation uses **three specialized CrewAI agents** running through a sequential process.

```text
CUSTOMER QUERY
      │
      ▼
┌─────────────────────────┐
│ Input Guardrails        │
│ Security Validation     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 1. Assistant Agent      │
│ Initial Answer          │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 2. Web Search Agent     │
│ Retrieve + Ground       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ 3. Entry Agent          │
│ Consolidate Record      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Output Guardrail        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Evaluation & Validation │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Persistent Record       │
│ answers.txt             │
└─────────────────────────┘
```

---

## Agent Responsibilities

### 1. Assistant Agent

The Assistant Agent:

- Understands the customer request
- Produces an initial response
- Uses model knowledge
- Does not claim that web verification has occurred

### 2. Web Search Agent

The Web Search Agent:

- Retrieves current information
- Uses web evidence
- Grounds the response
- Produces a current-information answer where required

### 3. Entry Agent

The Entry Agent:

- Consolidates the customer query
- Uses the grounded answer
- Creates a structured support record
- Prepares the record for validation and persistence

---

# 🧩 Core Capabilities

| Capability | Implementation |
|---|---|
| Multi-Agent Framework | CrewAI |
| Agents | 3 Specialized Agents |
| Orchestration | Sequential |
| LLM | OpenAI |
| Current Information | Web Search |
| Grounded Response | Search evidence supplied to research stage |
| Input Protection | Deterministic + AI-assisted validation |
| Output Protection | Post-generation Output Guardrail |
| Current-Date Correctness | Temporal Validation |
| Persistence | `answers.txt` |
| Session Context | Session-scoped memory |
| UI | Streamlit |
| Secrets | Environment Variables |
| Containerization | Docker |
| Reverse Proxy | Traefik |
| TLS | Let's Encrypt |
| Hosting | Hostinger VPS |

---

# 🛡️ Guardrails

Guardrails were implemented around the AI workflow instead of relying only on model prompts.

---

## Input Guardrails

Every request passes through a security gate before downstream execution.

```text
Customer Request
       │
       ▼
Deterministic Validation
       │
       ▼
AI Safety Check
       │
   ┌───┴───┐
   │       │
 PASS    BLOCK
   │       │
   ▼       ▼
Workflow  Stop Safely
```

This allows unsafe or invalid requests to be stopped before:

- Agents execute
- Search is invoked
- Tools are invoked
- Evaluation runs
- Data is persisted

---

## Output Guardrails

Generated content is also validated before being released.

```text
Generated Response
       │
       ▼
Output Guardrail
       │
   ┌───┴───┐
   │       │
 PASS    BLOCK
   │       │
   ▼       ▼
Display   Safe Message
Persist   No Persistence
```

The same post-generation guardrail principle is applied to both architectures used in **Experiment 1**.

This keeps the Python vs CrewAI comparison consistent.

---

# 📅 Temporal Validation

Current-information questions need more than semantic relevance.

Additional deterministic validation is applied to freshness-sensitive requests containing terms such as:

- `latest`
- `current`
- `newest`
- `today`
- `next`
- `upcoming`
- `nearest`

This helps prevent an outdated or past date from silently being presented as the latest or upcoming answer.

---

# 🧠 Session Memory

The Buildathon page supports **session-scoped conversational memory**.

Example:

```text
User: What is the latest stable version of Python?

User: When was it released?
```

The second question can use the context established by the first question.

## Memory Design

- Stored using Streamlit session state
- Limited to recent completed turns
- No separate memory-model/API call
- Can be cleared from the UI
- Scoped to the current application session
- Not implemented as a global shared memory store

This provides useful conversational continuity while controlling token growth.

---

# 🧪 Experiment 1 — Fixed Sequential Workflow

## Objective

Experiment 1 asks:

> **If the execution path is already known, do we really need an agent framework?**

Two implementations provide equivalent business capability.

---

## 🟢 Non-Agentic · Python Orchestration

```text
Customer Query
      │
      ▼
Explicit Python Orchestration
      │
      ▼
Initial OpenAI Answer
      │
      ▼
Shared Web Evidence
      │
      ▼
Grounded Synthesis
      │
      ▼
Output Guardrail
      │
      ▼
Validation
      │
      ▼
Persistence
```

The developer explicitly controls the sequence.

---

## 🔵 Agentic · CrewAI

```text
Customer Query
      │
      ▼
CrewAI Sequential Process
      │
      ▼
Assistant Agent
      │
      ▼
Web Search Agent
      │
      ▼
Entry Agent
      │
      ▼
Output Guardrail
      │
      ▼
Validation
      │
      ▼
Persistence
```

CrewAI provides role-based orchestration abstraction.

---

## Fair Comparison Design

Both architectures receive the **same workflow evidence** during a comparison run.

This avoids comparing two implementations using different search results.

A separate evidence set is retrieved for evaluation so that:

```text
Generation Evidence
        ≠
Independent Evaluation Evidence
```

This provides a stronger basis for comparison.

---

## Experiment 1 Evaluation

Both implementations are evaluated using common dimensions:

- **Relevance**
- **Completeness**
- **Consistency**
- **Groundedness**

The application also records:

- Execution time
- Search usage
- Known API-call information
- Persistence status
- Validation results

---

## Experiment 1 Learning

A fixed sequential workflow does not automatically require an agent framework.

Explicit Python may be attractive when:

- The execution path is predetermined
- Branching is minimal
- Tool choice is already known
- Deterministic control is preferred

CrewAI still provides useful:

- Role separation
- Agent abstraction
- Task abstraction
- Orchestration structure

However, that abstraction also introduces framework and runtime overhead.

> **Learning:** Do not introduce agents simply because multiple processing stages exist.

---

# 🧭 Experiment 2 — Dynamic Decision Workflow

Experiment 2 changes the problem.

Instead of comparing two implementations following a fixed sequence, the request may require different capabilities depending on customer intent.

---

## Explicit Python Routing

```text
Customer Request
       │
       ▼
Developer-Written Rules
       │
       ▼
Intent Matching
       │
       ▼
Invoke Matching Capabilities
       │
       ▼
Deterministic Response
```

Decision ownership remains in developer-written code.

---

## Runtime Agent Selection

```text
Customer Request
       │
       ▼
Runtime Interpretation
       │
       ▼
Select Relevant Capabilities
       │
       ▼
Execute Selected Tools
       │
       ▼
Customer Response
```

The agent chooses among a **bounded set of available capabilities**.

---

# 🔧 Bounded Capabilities

Experiment 2 exposes controlled simulated enterprise capabilities.

The tools intentionally avoid real external side effects.

The goal is to study:

- Runtime interpretation
- Capability selection
- Tool orchestration
- Decision ownership
- Execution traces

It is **not** intended to perform real billing or account operations.

---

# ✅ Decision Validation

Experiment 2 uses deterministic validation based on actual execution traces.

Checks include:

- Python routing trace completeness
- Allowed capability enforcement
- Tool-trace integrity
- Action-claim validation
- Usable runtime response
- Selection agreement between approaches

Decision validation uses actual runtime information and requires:

```text
0 additional OpenAI calls
0 additional search calls
```

---

# 💡 Experiment 2 Learning

Agents become more useful when:

- Requests vary
- The required execution path is not known beforehand
- Multiple bounded tools are available
- Different capability combinations may be required
- Developer-written routing logic starts growing

This does **not** mean ordinary Python cannot solve the problem.

The important difference is:

> **Where is the decision made?**

### Python

```text
Developer-written rules decide
```

### Agent

```text
Runtime reasoning selects among bounded capabilities
```

---

# 📊 Evaluation Strategy

Evaluation was treated as a separate engineering concern rather than assuming that a plausible-looking answer is correct.

---

## Fixed Workflow Evaluation

Useful dimensions include:

- Relevance
- Completeness
- Consistency
- Groundedness
- Temporal correctness

---

## Dynamic Workflow Evaluation

Answer quality alone is insufficient.

The system also needs to understand:

- Which capabilities were selected?
- Which capabilities actually executed?
- Were those capabilities allowed?
- Do claimed actions match the execution trace?
- Was a usable customer-facing response produced?

This is why the dynamic experiment presents:

## Decision Validation

rather than treating every evaluation problem as a generic LLM score.

---

# ⚡ API & Search Efficiency

One goal of the project was to avoid unnecessary API usage.

Implemented optimizations include:

- Shared workflow evidence in Experiment 1
- Independent but reusable evaluation evidence
- Session-level search caching
- Exact-result reuse
- Conservative near-duplicate evidence reuse
- Date-scoped cache behavior
- Session memory without an additional memory API
- Deterministic validation where an LLM is unnecessary
- Zero external API calls from Experiment 2 simulated tools
- Zero additional OpenAI/search calls for deterministic Decision Validation
- Learning Summary generated from local/session data

> **Use model calls where reasoning is useful. Keep deterministic work deterministic.**

---

# 🔭 Observability

The application exposes runtime information to make workflow behavior easier to understand and debug.

Visible information includes:

- Animated processing stages
- Current workflow stage
- Architecture comparison
- Execution time
- Known direct OpenAI call counts
- Workflow search-call counts
- Evaluation search-call counts
- Input Guardrail status
- Output Guardrail status
- Temporal Validation
- Tool traces
- Decision Validation
- Evaluation metrics
- Persistence status

CrewAI-controlled internal model calls are distinguished from application-controlled direct calls rather than presenting an invented exact count.

---

# 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Agent Framework | CrewAI |
| LLM | OpenAI |
| Search | Serper-backed Web Search |
| UI | Streamlit |
| Data Handling | Pandas |
| Configuration | python-dotenv |
| Persistence | Text File |
| Containerization | Docker |
| Container Orchestration | Docker Compose |
| Reverse Proxy | Traefik |
| TLS | Let's Encrypt |
| Hosting | Hostinger VPS |
| Source Control | GitHub |

---

# 📁 Repository Structure

```text
CrewAI-Customer-Support/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── .dockerignore
├── .gitignore
├── .env.example
│
└── assets/
    ├── experiment1_architecture.png
    └── experiment2_architecture.png
```

Runtime files such as `.env` and generated answer records should remain outside source control.

---

# 🔐 Environment Variables

Create a local `.env` file using `.env.example`.

Example:

```env
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

> **Security:** Never commit the real `.env`, API keys, GitHub PATs or other credentials.

Only placeholder variable names should be stored in `.env.example`.

---

# 💻 Running Locally

## Step 1 — Clone Repository

```bash
git clone https://github.com/arun-srinivasan-builds/CrewAI-Customer-Support.git

cd CrewAI-Customer-Support
```

---

## Step 2 — Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Configure Environment

Create:

```text
.env
```

and provide the required API keys.

---

## Step 5 — Start Streamlit

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🐳 Docker Deployment

## Build Image

```bash
docker build -t crewai-customer-support:1.0 .
```

---

## Run Container

If host port `8501` is already occupied:

```bash
docker run \
  --env-file .env \
  -p 8502:8501 \
  --name crewai-customer-support \
  crewai-customer-support:1.0
```

Open:

```text
http://localhost:8502
```

The application includes a Docker health check so container health can be verified independently of browser access.

---

# 🚀 VPS Deployment

The application is deployed using the following architecture:

```text
GitHub
   │
   ▼
Hostinger VPS
   │
   ▼
Docker Compose
   │
   ▼
CrewAI + Streamlit Container
   │
   ▼
Traefik Reverse Proxy
   │
   ▼
Let's Encrypt TLS
   │
   ▼
Public HTTPS Application
```

---

# 🌐 Live Deployment

## Application

### https://crewai-support.srv1965124.hstgr.cloud

The deployed application was functionally tested after deployment.

---

# 📦 VPS Deployment Steps

Clone the repository:

```bash
cd /docker/apps

git clone https://github.com/arun-srinivasan-builds/CrewAI-Customer-Support.git

cd CrewAI-Customer-Support
```

Create the production `.env` directly on the VPS.

The real `.env` is intentionally excluded from Git.

Build:

```bash
docker compose build
```

Start:

```bash
docker compose up -d
```

---

# 🩺 Container Health Verification

```bash
docker ps --filter name=crewai-customer-support
```

Check health:

```bash
docker inspect \
  --format='{{.State.Health.Status}}' \
  crewai-customer-support
```

Expected:

```text
healthy
```

---

# 🔒 HTTPS Verification

```bash
curl -I https://crewai-support.srv1965124.hstgr.cloud
```

Validated response:

```text
HTTP/2 200
```

---

# 🌐 Traefik + Let's Encrypt

Traefik routes:

```text
crewai-support.srv1965124.hstgr.cloud
```

to Streamlit on the application's internal container port:

```text
8501
```

Deployment validation confirmed:

- IPv4 DNS resolution
- IPv6 DNS resolution
- Docker network connectivity
- Traefik router configuration
- ACME certificate resolver
- Valid Let's Encrypt certificate
- Healthy application container
- HTTPS connectivity
- `HTTP/2 200`

---

# 🧪 Testing & Validation

| Test | Result |
|---|---|
| Streamlit Local Startup | ✅ PASS |
| Three-Agent Buildathon Flow | ✅ PASS |
| Web-Grounded Response | ✅ PASS |
| Entry Record Creation | ✅ PASS |
| Input Guardrail | ✅ PASS |
| Output Guardrail | ✅ PASS |
| Temporal Validation | ✅ PASS |
| Session Memory Follow-Up | ✅ PASS |
| Experiment 1 — Python Path | ✅ PASS |
| Experiment 1 — CrewAI Path | ✅ PASS |
| Experiment 1 Common Evaluation | ✅ PASS |
| Experiment 2 — Python Routing | ✅ PASS |
| Experiment 2 — Runtime Tool Selection | ✅ PASS |
| Decision / Action Validation | ✅ PASS |
| Docker Image Build | ✅ PASS |
| Docker Health Check | ✅ PASS |
| VPS Container Deployment | ✅ PASS |
| Traefik Routing | ✅ PASS |
| Let's Encrypt HTTPS | ✅ PASS |
| Public HTTP/2 Response | ✅ PASS |
| Live Browser Functional Test | ✅ PASS |

---

# 🧯 Issues Encountered & Resolutions

## Issue 1 — Docker Port Conflict

### Problem

Port `8501` was already used by another Streamlit container.

```text
Bind for 0.0.0.0:8501 failed: port is already allocated
```

### Resolution

The CrewAI application was mapped to host port:

```text
8502
```

while Streamlit continued using container port:

```text
8501
```

### Learning

Container ports and host ports are independent.

Multiple Streamlit applications can run simultaneously as long as host-port mappings do not conflict.

---

# 🧯 Issue 2 — GitHub Push Returned HTTP 403

### Problem

Git Credential Manager was authenticating using a different GitHub identity.

The push returned:

```text
Permission denied
HTTP 403
```

### Resolution

Repository authentication was corrected using a **fine-grained GitHub Personal Access Token**.

The token was restricted to the repository with:

```text
Contents → Read and Write
```

### Learning

Git commit identity:

```text
user.name
user.email
```

and GitHub authentication are separate concerns.

---

# 🧯 Issue 3 — VPS Git Commit Used Root Identity

### Problem

A deployment commit created on the VPS initially inherited the Linux root identity.

### Resolution

Git identity was configured correctly:

```bash
git config --global user.name "arun-srinivasan-builds"

git config --global user.email "<configured email>"
```

The commit author was then corrected before push.

### Learning

Configure Git author information on deployment machines before creating commits.

---

# 🧯 Issue 4 — Remote Branch Ahead of Laptop

### Problem

A laptop push was rejected because deployment commits already existed on:

```text
origin/main
```

### Resolution

The local commit was rebased:

```bash
git pull --rebase origin main
```

and then pushed normally.

### Learning

Use a consistent development/deployment flow:

```text
Laptop Development
       │
       ▼
GitHub
       │
       ▼
VPS git pull
       │
       ▼
Docker Rebuild
```

Avoid modifying application source directly on the VPS unless necessary.

---

# 🧯 Issue 5 — HTTPS Initially Returned Self-Signed Certificate

### Problem

Immediately after configuring the new hostname, HTTPS temporarily returned a certificate verification error.

### Investigation

The following were independently checked:

- Traefik labels
- Docker networks
- IPv4 DNS
- IPv6 DNS
- ACME resolver configuration
- Application health

### Resolution

Traefik successfully completed Let's Encrypt certificate issuance.

Final certificate verification confirmed a valid Let's Encrypt certificate.

The endpoint returned:

```text
HTTP/2 200
```

### Learning

DNS, application health, routing and TLS should be diagnosed as separate infrastructure layers.

---

# 🧯 Issue 6 — Experiment 1 `entry_record_display` Error

### Problem

After VPS deployment, the Non-Agentic Experiment 1 path raised:

```text
NameError: name 'entry_record_display' is not defined
```

### Root Cause

The Python path referenced Output Guardrail display variables that had been implemented in the Agentic path but were not initialized consistently in the Non-Agentic path.

### Resolution

The Non-Agentic implementation was aligned with the same post-generation Output Guardrail behavior.

```text
Generate
   │
   ▼
Ground
   │
   ▼
Output Guardrail
   │
 ┌─┴────────────┐
 │              │
PASS          BLOCK
 │              │
 ▼              ▼
Display       Safe Message
Persist       No Persistence
```

The corrected application was:

1. Tested locally
2. Committed to GitHub
3. Pulled onto the VPS
4. Docker image rebuilt
5. Container restarted
6. Health checked
7. Public HTTPS endpoint validated
8. Functionally tested again

### Learning

Guardrails must be integrated consistently across every execution path.

---

# 🔒 Security Practices

The project follows practical secret and application-security controls.

- `.env` excluded from Git
- `.env.example` contains placeholders only
- API keys supplied through environment variables
- GitHub PAT excluded from the repository
- Fine-grained repository access used for Git authentication
- Deterministic input validation
- AI-assisted safety validation
- Post-generation Output Guardrail
- Blocked output is not persisted
- Dynamic-agent capabilities are bounded
- Action claims are validated against actual tool traces
- Production application is exposed through Traefik rather than a direct public container port

---

# 💡 Key Learning Outcomes

## 1. Agents Are Not Automatically Better

A predictable fixed workflow can often be implemented more simply using explicit Python orchestration.

---

## 2. Runtime Uncertainty Changes the Equation

Agent-based orchestration becomes more interesting when the required capability combination varies depending on the request.

---

## 3. Grounding Is Different from Generation

A fluent LLM response is not automatically grounded.

Evidence retrieval, synthesis constraints and evaluation need to be designed explicitly.

---

## 4. Guardrails Belong Outside the Prompt

Prompt instructions are useful, but security should not depend entirely on asking the model to behave correctly.

Application-level controls provide additional protection.

---

## 5. Evaluation Must Match the Problem

A factual fixed workflow benefits from:

```text
Relevance
Completeness
Consistency
Groundedness
```

A dynamic tool-selection workflow also requires:

```text
Decision Validation
Tool Trace Validation
Action Validation
```

---

## 6. Memory Has a Cost

Session memory improves follow-up conversations but increases prompt context.

Bounded memory provides a practical balance.

---

## 7. API Efficiency Matters

Agents, search, evaluation and memory can multiply API usage quickly.

Useful optimizations include:

```text
Caching
Shared Evidence
Exact Result Reuse
Deterministic Validation
Bounded Memory
```

---

## 8. Deployment Is Part of the Product

A successful local application is only one stage.

Production-style delivery also involves:

- Docker
- Health checks
- Secret management
- Git workflow
- VPS deployment
- DNS
- Reverse proxy
- TLS
- Live functional validation

---

# 🎯 Final Takeaway

The most important lesson from this Buildathon was not simply how to create multiple agents.

It was understanding **where agents provide meaningful value and where ordinary deterministic code remains sufficient**.

```text
Is the execution path predictable?
              │
       ┌──────┴──────┐
       │             │
      YES            NO
       │             │
       ▼             ▼
Explicit         Runtime
Orchestration    Reasoning
may be           may add
sufficient       more value
```

The goal is **not to maximize the number of agents**.

The goal is to choose the appropriate orchestration approach for the problem.

> **Use agents deliberately. Keep deterministic work deterministic. Introduce runtime agent reasoning where it provides meaningful value.**

---

# 🚀 Live Demo

## AI Customer Support Lab

### https://crewai-support.srv1965124.hstgr.cloud

**Dockerized · VPS Hosted · Traefik Routed · HTTPS Secured**

---

# 📂 GitHub Repository

### https://github.com/arun-srinivasan-builds/CrewAI-Customer-Support

---

# 👤 Author

## Arun Srinivasan

Hands-on Generative AI learning through:

**Building · Experimentation · Validation · Deployment**
