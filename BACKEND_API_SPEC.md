# Siri Multi-Agent Backend — API & Frontend Integration Specification

> **Purpose:** This document provides a complete technical specification of the Siri Multi-Agent backend API. Provide this file to an AI agent or frontend developer to build a modern, high-performance web frontend (e.g., React, Next.js, Vue, Svelte, or Vanilla JS/Tailwind) for this backend.

---

## 1. System Overview & Architecture

**Siri Multi-Agent Assistant** is an intelligent conversational platform built on **FastAPI** and **LangGraph** with durable **Semantic Memory** (Pinecone vector store + background distillation) and multi-agent routing.

### Key Capabilities for the Frontend:
1. **Multi-Agent Orchestration**: Requests pass through a graph of specialized agents:
   - `guardrail`: Moderation and safety checking.
   - `supervisor`: Master planner and request classifier.
   - `sub_infosupervisor` & `sub_actionsupervisor`: Domain coordinators.
   - `weather`: Real-time weather lookup.
   - `researcher`: Web search and knowledge retrieval.
   - `action`: Terminal, system commands, and tool execution.
   - `output_sanitizer`: Final response sanitization and formatting.
2. **Synchronous & SSE Streaming**: Support for standard REST `/chat` and real-time Server-Sent Events (SSE) `/chat/stream` showing live node/agent transitions ("hops").
3. **Long-Term Semantic Memory**: Scoped per `user_id`, stores learned facts and injects personalized profiles into conversations.

---

## 2. Server Configuration & Base URL

| Setting | Default Value | Notes |
| :--- | :--- | :--- |
| **Local Base URL** | `http://localhost:8000` or `http://127.0.0.1:8000` | FastAPI server with Uvicorn |
| **Interactive Docs (Swagger UI)** | `http://localhost:8000/docs` | OpenAPI specification viewer |
| **Alternative Docs (ReDoc)** | `http://localhost:8000/redoc` | Clean document view |
| **OpenAPI JSON Schema** | `http://localhost:8000/openapi.json` | Raw OpenAPI 3.1.0 schema |
| **CORS Policy** | `allow_origins=["*"]` | Pre-configured to allow all origins, methods, and headers with credentials |
| **Authentication** | None required (open REST) | Ready for direct client API calls |
| **Default User ID** | `siri_user` | Used for memory scoping if not specified |

---

## 3. TypeScript Type Definitions

Use these TypeScript interfaces for frontend state management and API clients:

```typescript
// ==========================
// Chat Types
// ==========================

export type MessageRole = 'user' | 'human' | 'assistant' | 'ai' | 'system';

export interface ChatMessage {
  role: MessageRole;
  content: string;
}

/**
 * Request payload for /chat and /chat/stream.
 * Provide EITHER 'message' (single query) OR 'messages' (full history).
 */
export interface ChatRequest {
  message?: string;
  messages?: ChatMessage[];
  user_id?: string; // Defaults to "siri_user"
}

export interface ChatResponse {
  response: string;    // Final response text from the agent system
  hop_count: number;   // Total graph nodes/agents traversed
  user_id: string;     // Active user ID
  status: 'success' | string;
}

// ==========================
// SSE Stream Event Types
// ==========================

export interface SSEStartEvent {
  event: 'start';
  user_id: string;
}

export interface SSENodeUpdateEvent {
  event: 'node_update';
  hop: number;
  next: 'guardrail' | 'supervisor' | 'sub_infosupervisor' | 'sub_actionsupervisor' | 'weather' | 'researcher' | 'action' | 'output_sanitizer' | string;
}

export interface SSECompleteEvent {
  event: 'complete';
  response: string;
  hops: number;
}

export interface SSEErrorEvent {
  event: 'error';
  detail: string;
}

export type SSEStreamEvent = 
  | SSEStartEvent 
  | SSENodeUpdateEvent 
  | SSECompleteEvent 
  | SSEErrorEvent;

// ==========================
// Health & System Types
// ==========================

export interface HealthResponse {
  status: 'ok' | string;
  version: string;
  semantic_memory_enabled: boolean;
  default_user_id: string;
}

export interface RootResponse {
  name: string;
  version: string;
  docs_url: string;
  health_url: string;
}

// ==========================
// Semantic Memory Types
// ==========================

export interface MemoryFact {
  fact_id?: string;
  content: string;
  confidence?: number;
  category?: string;
  [key: string]: any;
}

export interface MemoryResponse {
  user_id: string;
  fact_count: number;
  facts: MemoryFact[];
  formatted_profile: string | null;
}

export interface ClearMemoryResponse {
  status: 'success' | string;
  message: string;
  cleared_facts: number;
  user_id: string;
}
```

---

## 4. API Endpoints Reference

### 4.1 System & Health

#### `GET /`
Returns service metadata and documentation links.

- **Request**: No parameters/body.
- **Headers**: `Accept: application/json`
- **Response `200 OK`**:
```json
{
  "name": "Siri Multi-Agent API",
  "version": "1.0.0",
  "docs_url": "/docs",
  "health_url": "/health"
}
```

---

#### `GET /health`
Inspects operational health and memory module status.

- **Request**: No parameters/body.
- **Headers**: `Accept: application/json`
- **Response `200 OK`**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "semantic_memory_enabled": true,
  "default_user_id": "siri_user"
}
```

---

### 4.2 Chat Endpoints

#### `POST /chat`
Synchronous turn execution. Sends user query or chat history and waits for the full multi-agent response.

- **Endpoint**: `/chat`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Request Body Options**:

*Option A (Single prompt query):*
```json
{
  "message": "What is the weather in Tokyo?",
  "user_id": "siri_user"
}
```

*Option B (Conversation history):*
```json
{
  "messages": [
    { "role": "user", "content": "My name is Jugal and I like TypeScript." },
    { "role": "assistant", "content": "Nice to meet you, Jugal!" },
    { "role": "user", "content": "What was my favorite language?" }
  ],
  "user_id": "siri_user"
}
```

- **Response `200 OK`**:
```json
{
  "response": "Your favorite language is TypeScript!",
  "hop_count": 4,
  "user_id": "siri_user",
  "status": "success"
}
```

- **Error Responses**:
  - `422 Unprocessable Entity`: Missing both `message` and `messages`.
  - `500 Internal Server Error`: Agent graph execution error (`{"detail": "Agent execution error: ..."}`).
  - `503 Service Unavailable`: Backend service not yet initialized.

---

#### `POST /chat/stream`
Real-time streaming endpoint utilizing **Server-Sent Events (SSE)**. Emits step-by-step agent graph transitions and the final message.

- **Endpoint**: `/chat/stream`
- **Method**: `POST`
- **Headers**:
  - `Content-Type: application/json`
  - `Accept: text/event-stream`
- **Request Body**: Same as `/chat` (`ChatRequest`).
- **Response Content-Type**: `text/event-stream; charset=utf-8`

#### SSE Stream Protocol Flow:
The stream yields newline-delimited `data: <json>\n\n` payloads:

1. **Start Event**: Emitted when execution begins.
```text
data: {"event": "start", "user_id": "siri_user"}

```

2. **Node Update Events**: Emitted each time an agent node is executed in LangGraph.
```text
data: {"event": "node_update", "hop": 1, "next": "guardrail"}

data: {"event": "node_update", "hop": 2, "next": "supervisor"}

data: {"event": "node_update", "hop": 3, "next": "sub_infosupervisor"}

data: {"event": "node_update", "hop": 4, "next": "weather"}

data: {"event": "node_update", "hop": 5, "next": "output_sanitizer"}

```

3. **Complete Event**: Emitted when graph execution finishes, containing the full response.
```text
data: {"event": "complete", "response": "The current weather in Tokyo is 22°C with clear skies.", "hops": 5}

```

4. **Error Event** (If an error occurs):
```text
data: {"event": "error", "detail": "API rate limit exceeded"}

```

---

### 4.3 Semantic Memory Endpoints

#### `GET /memory/{user_id}`
Fetches active long-term semantic memory facts and the formatted profile injected into agent prompts for a specific user.

- **Endpoint**: `/memory/{user_id}` (e.g., `/memory/siri_user`)
- **Method**: `GET`
- **Headers**: `Accept: application/json`
- **Response `200 OK`**:
```json
{
  "user_id": "siri_user",
  "fact_count": 2,
  "facts": [
    {
      "fact_id": "fact_01HXYZ...",
      "content": "User prefers concise answers with code snippets.",
      "confidence": 0.95,
      "category": "preference"
    },
    {
      "fact_id": "fact_02HABC...",
      "content": "User is building a web dashboard in React.",
      "confidence": 0.9,
      "category": "project"
    }
  ],
  "formatted_profile": "- User prefers concise answers with code snippets.\n- User is building a web dashboard in React."
}
```

---

#### `DELETE /memory/{user_id}`
Clears all stored semantic memory facts and pending learning queues for a specific user.

- **Endpoint**: `/memory/{user_id}` (e.g., `/memory/siri_user`)
- **Method**: `DELETE`
- **Response `200 OK`**:
```json
{
  "status": "success",
  "message": "Memory successfully cleared for user 'siri_user'",
  "cleared_facts": 2,
  "user_id": "siri_user"
}
```

---

#### `DELETE /memory`
Clears semantic memory for the default user (`siri_user`).

- **Endpoint**: `/memory`
- **Method**: `DELETE`
- **Response `200 OK`**: Same format as `DELETE /memory/{user_id}`.

---

## 5. Agent Flow & Execution Visualization

When building the UI, you can display the real-time agent execution pipeline. Here is the multi-agent hierarchy:

```
[ User Input ]
      │
      ▼
┌──────────────┐
│  guardrail   │  ── (Safety & Moderation Check)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  supervisor  │  ── (Master Routing & Task Planning)
└──────┬───────┘
       ├────────────────────────────────┐
       ▼                                ▼
┌──────────────────────┐     ┌──────────────────────┐
│ sub_infosupervisor   │     │ sub_actionsupervisor │
└──────┬───────────────┘     └──────────┬───────────┘
       ├──────────────┐                 │
       ▼              ▼                 ▼
┌─────────────┐ ┌────────────┐   ┌──────────────┐
│  researcher │ │  weather   │   │ action/shell │
└─────────────┘ └────────────┘   └──────────────┘
       │              │                 │
       └──────────────┴─────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ output_sanitizer │  ── (Final Formatting & Cleansing)
            └─────────┬────────┘
                      │
                      ▼
              [ Final Output ]
```

### Agent Badge / Node Display Guide:
- **`guardrail`**: Shield icon / "Safety Verification"
- **`supervisor`**: Compass icon / "Master Supervisor"
- **`sub_infosupervisor`**: Info/Brain icon / "Information Coordinator"
- **`sub_actionsupervisor`**: Sliders/Gears icon / "Action Coordinator"
- **`weather`**: Cloud/Sun icon / "Weather Tool"
- **`researcher`**: Search/Book icon / "Research & Knowledge Agent"
- **`action`**: Terminal icon / "System Action Agent"
- **`output_sanitizer`**: Sparkles/Filter icon / "Output Sanitizer"

---

## 6. Ready-to-Use Frontend Integration Code

### 6.1 Standard Chat (Axios / Fetch)

```typescript
const API_BASE = "http://localhost:8000";

export async function sendMessage(message: string, userId: string = "siri_user") {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, user_id: userId }),
  });

  if (!response.ok) {
    throw new Error(`Chat error: ${response.statusText}`);
  }

  const data: ChatResponse = await response.json();
  return data;
}
```

### 6.2 SSE Streaming Client with Live Node Updates

```typescript
const API_BASE = "http://localhost:8000";

interface StreamCallbacks {
  onStart?: (userId: string) => void;
  onNodeUpdate?: (hop: number, node: string) => void;
  onComplete?: (response: string, hops: number) => void;
  onError?: (error: string) => void;
}

export async function streamChat(
  message: string,
  userId: string = "siri_user",
  callbacks: StreamCallbacks
) {
  try {
    const response = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
      },
      body: JSON.stringify({ message, user_id: userId }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`Failed to initiate stream: ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          const jsonStr = trimmed.replace("data: ", "").trim();
          if (!jsonStr) continue;

          try {
            const data: SSEStreamEvent = JSON.parse(jsonStr);
            if (data.event === "start") {
              callbacks.onStart?.(data.user_id);
            } else if (data.event === "node_update") {
              callbacks.onNodeUpdate?.(data.hop, data.next);
            } else if (data.event === "complete") {
              callbacks.onComplete?.(data.response, data.hops);
            } else if (data.event === "error") {
              callbacks.onError?.(data.detail);
            }
          } catch (err) {
            console.error("Failed to parse SSE JSON:", jsonStr, err);
          }
        }
      }
    }
  } catch (err: any) {
    callbacks.onError?.(err.message || "Network error");
  }
}
```

### 6.3 Memory Management API Calls

```typescript
// Fetch user's memory facts & profile
export async function getMemory(userId: string = "siri_user"): Promise<MemoryResponse> {
  const res = await fetch(`${API_BASE}/memory/${encodeURIComponent(userId)}`);
  return res.json();
}

// Clear user's memory
export async function clearMemory(userId: string = "siri_user"): Promise<ClearMemoryResponse> {
  const res = await fetch(`${API_BASE}/memory/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
  return res.json();
}
```

---

## 7. Recommended Frontend UI Features

To create a visually impressive and intuitive interface, consider implementing:

1. **Main Chat Workspace**:
   - Modern conversation view with Markdown formatting (code blocks, tables, lists).
   - User vs. Assistant message bubbles with distinct avatars.
   - Quick action suggestions (e.g., *"What is the weather in Paris?"*, *"Research quantum computing"*, *"Remember that I prefer dark mode"*).
2. **Live Agent Execution Tracker (Hop Visualizer)**:
   - When using `/chat/stream`, show an animated step-by-step progress pill or pipeline stepper showing each node as it executes (`guardrail` ➔ `supervisor` ➔ `weather` ➔ `output_sanitizer`).
   - Display total hops executed upon completion.
3. **Semantic Memory Drawer / Sidebar**:
   - Side panel showing active user's remembered facts.
   - Confidence score indicators / tags.
   - "Clear Memory" button with confirmation modal.
4. **User Switcher & Session Scoping**:
   - Dropdown or avatar selector to switch `user_id` (e.g., `siri_user`, `developer_1`, `guest`) to demonstrate personalized memory per user.
5. **Backend Health Indicator**:
   - Live pill in navbar querying `GET /health` (Green: Online, Red: Offline, Semantic Memory: Enabled/Disabled badge).
