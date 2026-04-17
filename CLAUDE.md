# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered fashion recommendation system built on top of the **Gorse** open-source recommendation engine. It has two main subsystems:

1. **Gorse core** (root of repo) — distributed recommendation engine (Master/Server/Worker nodes) written in Go
2. **fashion-recommend/** — a fashion-domain API layer built with Gin that integrates Gorse with LLM capabilities

## Common Commands

### fashion-recommend (primary development area)

```bash
cd fashion-recommend

# Run API server
make run
# or directly:
go run main.go

# Method 3: flag-based startup (explicit port/db/gorse overrides)
go run main.go \
  -port 5001 \
  -db "host=localhost port=5432 user=gorse password=gorse_pass dbname=gorse sslmode=disable" \
  -gorse "http://localhost:8088"

# Build binaries
make build              # outputs to bin/fashion-api and bin/init-data

# Run tests
make test
# or a single package:
go test -v ./api/...

# Initialize sample data
make init-data
# or: go run data/init_data.go

# Go proxy (China mainland users — set before downloading dependencies)
export GOPROXY=https://goproxy.cn,direct
go mod download
go mod verify

# Start all infrastructure (Postgres, Redis, Gorse nodes)
make docker-up

# Stop infrastructure
make docker-down
```

### Frontend (fashion-recommend/frontend/)

```bash
cd fashion-recommend/frontend

npm install
# Note: vite.config.ts shoproxy was manually updated to include /images (not in original template):
#   '/api'    → http://localhost:5001
#   '/images' → http://localhost:5001
npm run dev       # dev server
npm run build     # build to dist/ (served by Go backend)
npm run preview
```

### Admin Dashboard (admin-dashboard/)

```bash
cd admin-dashboard

# Install dependencies
npm install

# Build frontend
npm run build     # outputs to dist/

# Start server (runs on http://localhost:3001)
node server.js

# Or use PM2 for persistent process management (recommended)
npm install -g pm2
pm2 start server.js --name admin-dashboard
pm2 status
pm2 logs admin-dashboard
pm2 stop admin-dashboard
pm2 restart admin-dashboard
```

### Gorse Core (root module)

```bash
# Build all Gorse binaries
go build ./cmd/...

# Run tests
go test ./...

# Run a specific package test
go test -v ./logics/...
```

## Environment Variables (fashion-recommend)

| Variable | Default | Description |
|---|---|---|
| `GORSE_ENDPOINT` | `http://localhost:8088` | Gorse master HTTP endpoint |
| `GORSE_API_KEY` | `` | Gorse API key |
| `PORT` | `5001` | API server port |
| `DATABASE_URL` | `host=localhost port=5432 user=gorse password=gorse_pass dbname=gorse sslmode=disable` | PostgreSQL connection string |
| `AI_API_KEY` | (Aliyun DashScope key) | LLM API key |
| `AI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible LLM endpoint |
| `AI_MODEL` | `qwen-plus` | LLM model name |

## Architecture

### Gorse Distributed Nodes

The Gorse core runs as three separate processes that communicate via gRPC (port 8086):

- **Master** (`master/`, port 8086 gRPC / 8088 HTTP) — orchestrates the system, trains CF and CTR models, serves the admin dashboard
- **Server** (`server/`, port 8087 HTTP) — serves REST recommendation APIs to clients
- **Worker** (`worker/`, port 8089 HTTP) — executes background recommendation jobs

The Master trains models and pushes them to Server/Worker nodes. All inter-node communication uses Protocol Buffers defined in `protocol/`.

### Recommendation Algorithm Pipeline (`logics/`)

Recommendations are built by chaining multiple algorithms with fallbacks:

1. **Collaborative Filtering** (`cf.go`) — user-item interaction matrix factorization
2. **Item-to-Item** (`item_to_item.go`) — content similarity
3. **User-to-User** (`user_to_user.go`) — social similarity
4. **Non-personalized** (`non_personalized.go`) — trending, popular, new arrivals
5. **LLM Re-ranking** (`chat.go`) — LLM-based reranking and explanation generation

### fashion-recommend Service (`fashion-recommend/`)

A Gin-based HTTP API that sits in front of Gorse. Key packages:

- `api/` — route handlers organized by domain (auth, AI, comments, likes, items, users, recommendations)
- `ai/` — OpenAI-compatible client (default: Aliyun DashScope qwen-plus)
- `client/` — HTTP client for Gorse master/server APIs
- `traits/` — LLM-powered user style preference extraction; syncs traits back to Gorse as user labels
- `database/` — PostgreSQL models for conversations, messages, user traits, and social interactions (comments, likes)
- `auth/` — session-based auth service
- `models/` — shared data models (User, Item, Feedback)

The frontend (`fashion-recommend/frontend/`) is a React/TypeScript/Vite/Tailwind SPA. Its production build goes to `frontend/dist/` and is served as static files by the Go backend via Gin.

### Storage Layer

Gorse supports pluggable backends configured in `config/config.toml`:

- **Data store**: MySQL, PostgreSQL, MongoDB, ClickHouse, or SQLite
- **Cache store**: Redis or MySQL
- **Vector store** (`storage/vectors/`): Milvus, Qdrant, Weaviate, or SQLite — used for ANN (approximate nearest-neighbor) search powering item-to-item and user-to-user similarity
- **fashion-recommend** uses PostgreSQL directly (via `database/` package) for social features not managed by Gorse

#### Vector Store (`storage/vectors/`)

All vector backends implement a common `Database` interface (`database.go`) with these operations: `AddCollection`, `DeleteCollection`, `AddVectors`, `DeleteVectors`, `QueryVectors`. The backend is selected by URI prefix (`milvus://`, `qdrant://`, `weaviate://`, `sqlite://`).

The **Milvus** backend (`milvus.go`) is the most full-featured: it uses the official `milvus-sdk-go/v2` and manages schema creation (id, vector, categories array, timestamp fields), HNSW index creation with configurable distance metrics (Cosine/L2/IP), and filtered ANN search using Milvus expression syntax (e.g., `array_contains(categories, 'X')`). The `proxy.go` file wraps any backend with caching/proxy logic.

### LLM Integration Points

- `logics/chat.go` — LLM re-ranking within Gorse core (uses Ollama/qwen2.5 by default per `config/config.toml`)
- `fashion-recommend/ai/service.go` — chat, recommendation explanation, style advice (uses Aliyun DashScope qwen-plus)
- `fashion-recommend/traits/extractor.go` — extracts structured style/color/occasion preferences from user text; maps Chinese keywords to English Gorse labels

### Data Flow

```
User actions (comments, likes) → fashion-recommend API
    → AI trait extraction → Gorse (user labels)
    → Gorse recommendation algorithms (CF, item-to-item, etc.)
    → Cached recommendations → fashion-recommend API → Frontend
```

## Database Setup (Manual — without Docker)

### Install PostgreSQL

macOS:
```bash
brew install postgresql@15
brew services start postgresql@15
# Stop: brew services stop postgresql@15
```

Ubuntu/Debian:
```bash
sudo apt update && sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Create user and database
```sql
-- run: psql postgres
CREATE USER gorse WITH PASSWORD 'gorse_pass';
CREATE DATABASE gorse OWNER gorse;
GRANT ALL PRIVILEGES ON DATABASE gorse TO gorse;
\q
```

### Verify connection
```bash
psql -h localhost -U gorse -d gorse -W
# password: gorse_pass
```

### Initialize tables (optional — auto-created on first run)
```sql
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE items (
  id SERIAL PRIMARY KEY,
  item_id VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  category VARCHAR(100),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE likes (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  item_id VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(user_id, item_id)
);
CREATE TABLE comments (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255) NOT NULL,
  item_id VARCHAR(255) NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Key Configuration Files

- `config/config.toml` — Gorse core configuration (DB, ports, algorithm parameters, LLM)
- `fashion-recommend/config/config.toml` — fashion subsystem configuration
- `fashion-recommend/docker-compose.yml` — spins up Postgres, Redis, and all three Gorse nodes