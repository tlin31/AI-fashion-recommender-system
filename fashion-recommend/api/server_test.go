package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"fashion-recommend/ai"
	"fashion-recommend/models"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// TestMain 关掉 gin 的 debug 路由表输出，否则每个用例都会刷 30 行日志，
// 真正的失败信息被淹没。
func TestMain(m *testing.M) {
	gin.SetMode(gin.TestMode)
	os.Exit(m.Run())
}

// newTestServer 构造一个只连 mock Gorse 的 Server。
//
// ai.Config 留空、db 传 nil：本文件的用例只覆盖直接转发到 Gorse 的路由
// （health / user / item / feedback / recommend / similar），这些路由不碰
// s.db 也不碰 s.aiService。需要数据库或 LLM 的路由（auth、comments、likes、
// ai-chat）要另起一组用例并注入真实依赖，不要在这里加。
func newTestServer(gorseURL string) *Server {
	return NewServer(gorseURL, "", ai.Config{}, nil)
}

// MockGorseServer 模拟 Gorse 服务器
func MockGorseServer() *httptest.Server {
	handler := http.NewServeMux()

	// 模拟插入用户
	handler.HandleFunc("/api/user", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"message": "success"})
		}
	})

	// 模拟获取用户
	handler.HandleFunc("/api/user/", func(w http.ResponseWriter, r *http.Request) {
		user := models.User{
			UserId: "test_user",
			Labels: []string{"gender:female", "style:casual"},
		}
		json.NewEncoder(w).Encode(user)
	})

	// 模拟批量插入用户
	handler.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"message": "success"})
	})

	// 模拟插入商品
	handler.HandleFunc("/api/item", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"message": "success"})
		}
	})

	// 模拟获取商品
	handler.HandleFunc("/api/item/", func(w http.ResponseWriter, r *http.Request) {
		item := models.Item{
			ItemId:     "test_item",
			Categories: []string{"women", "tops"},
			Labels:     []string{"brand:zara", "style:casual"},
			Timestamp:  time.Now(),
		}
		json.NewEncoder(w).Encode(item)
	})

	// 模拟批量插入商品
	handler.HandleFunc("/api/items", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"message": "success"})
	})

	// 模拟插入反馈
	handler.HandleFunc("/api/feedback", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"message": "success"})
	})

	// 模拟获取推荐
	//
	// 形状必须和真实 Gorse 一致：/api/recommend/{user-id} 默认返回**裸 item id 数组**
	// （server/rest.go:914），不是对象数组。这个 mock 原本返回 [{item_id, score}]，
	// 与 client.GetRecommend 的 []string 解析对不上，恒定 500——因为测试编译不过，
	// 这个偏差一直没被发现。
	handler.HandleFunc("/api/recommend/", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]string{"item_001", "item_002"})
	})

	// 模拟获取相似商品
	//
	// 与 recommend 相反，neighbors 走 SearchDocuments，返回 []cache.Score，
	// JSON 形状是 [{"Id": ..., "Score": ...}]（字段名首字母大写，无 json tag）。
	handler.HandleFunc("/api/item/test_item/neighbors", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode([]map[string]any{
			{"Id": "similar_001", "Score": 0.92},
			{"Id": "similar_002", "Score": 0.85},
		})
	})

	return httptest.NewServer(handler)
}

func TestHealthCheck(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/health", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "ok", response["status"])
	assert.Equal(t, "fashion-recommend-api", response["service"])
}

func TestCreateUser(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	user := models.NewFashionUser("test_user_001").
		WithGender("female").
		WithStyle("casual").
		Build()

	jsonData, _ := json.Marshal(user)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/user", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "user created successfully", response["message"])
	assert.Equal(t, "test_user_001", response["user_id"])
}

func TestCreateItem(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	item := models.NewFashionItem("test_item_001").
		WithCategories("women", "tops").
		WithBrand("zara").
		WithStyle("casual").
		Build()

	jsonData, _ := json.Marshal(item)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/item", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "item created successfully", response["message"])
	assert.Equal(t, "test_item_001", response["item_id"])
}

func TestCreateFeedback(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	feedbacks := []models.Feedback{
		{
			FeedbackType: "purchase",
			UserId:       "user_001",
			ItemId:       "item_001",
			Value:        1.0,
			Timestamp:    time.Now(),
		},
	}

	jsonData, _ := json.Marshal(feedbacks)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/feedback", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "feedback created successfully", response["message"])
}

func TestGetRecommend(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/recommend/user_001?n=10", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "user_001", response["user_id"])
	assert.NotNil(t, response["items"])
}

func TestGetSimilarItems(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/similar/test_item?n=10", nil)
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "test_item", response["item_id"])
	assert.NotNil(t, response["items"])

	// 回归断言：neighbors 返回的是 [{Id, Score}] 对象，client 必须解出 Gorse 给的
	// 真实分数。之前它按 []string 解析，既拿不到 id 也拿不到分数，整个接口 500。
	items, ok := response["items"].([]any)
	assert.True(t, ok, "items 应该是数组")
	assert.Len(t, items, 2)
	first := items[0].(map[string]any)
	assert.Equal(t, "similar_001", first["item_id"])
	assert.InDelta(t, 0.92, first["score"], 1e-9, "必须是 Gorse 的真实分数，不是按序号伪造的")
}

func TestCreateUsers_Batch(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	users := []models.User{
		models.NewFashionUser("user_001").WithGender("female").Build(),
		models.NewFashionUser("user_002").WithGender("male").Build(),
	}

	jsonData, _ := json.Marshal(users)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/users", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "users created successfully", response["message"])
	assert.Equal(t, float64(2), response["count"])
}

func TestCreateItems_Batch(t *testing.T) {
	mockServer := MockGorseServer()
	defer mockServer.Close()

	server := newTestServer(mockServer.URL)
	router := server.GetRouter()

	items := []models.Item{
		models.NewFashionItem("item_001").WithBrand("zara").Build(),
		models.NewFashionItem("item_002").WithBrand("uniqlo").Build(),
	}

	jsonData, _ := json.Marshal(items)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/items", bytes.NewBuffer(jsonData))
	req.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)
	assert.Equal(t, "items created successfully", response["message"])
	assert.Equal(t, float64(2), response["count"])
}
