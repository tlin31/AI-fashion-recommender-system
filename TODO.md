# TODO

## 🔬 Recommendation Model (`test/recommendation_model`)
- [ ] Implement GenerateImplicitFeedback in fashion-recommend/traits/gorse_sync.go
- [ ] Run and validate BPR vs ALS hyperparameter tuning (Goptuna, NDCG@10)
- [ ] Compare model scores against popularity baseline (target: NDCG 0.47)
- [ ] Evaluate item-to-item and user-to-user HNSW index quality

## 🌐 Frontend (`fashion-recommend/frontend/`)
- [ ] Complete AI Chat page implementation
- [ ] Integration test recommendation display with live Gorse API
- [ ] Refine ProductCard and CommentDrawer UX
- [ ] Run `npm run build` and verify static assets served by Go backend

## ⚙️ Backend API (`fashion-recommend/api/`)
- [ ] End-to-end test recommendation endpoint flow (traits → Gorse → response)
- [ ] Validate LLM trait extraction and sync back to Gorse user labels
- [ ] Review error handling across auth, items, likes, comments handlers

## 🧪 Testing
- [ ] Add integration tests for fashion-recommend API (`go test ./api/...`)
- [ ] Test Gorse CF model training pipeline end-to-end
- [ ] Frontend unit tests for key components

## 🗄️ Data & Infrastructure
- [ ] Run `make init-data` to seed PostgreSQL with sample fashion items
- [ ] Validate `make docker-up` (Postgres, Redis, Gorse nodes) starts cleanly
- [ ] Confirm Redis cache is populated after first recommendation run

## 🛠️ Admin Dashboard (`admin-dashboard/`)
- [ ] Clarify scope: custom dashboard vs Gorse built-in admin (port 8088)
- [ ] Initialize or integrate with existing Gorse master dashboard

## 📝 Documentation
- [ ] Document AI trait extraction keyword → Gorse label mappings
- [ ] Add API endpoint reference for fashion-recommend routes
