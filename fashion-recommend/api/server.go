package api

import (
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"

	"fashion-recommend/ai"
	"fashion-recommend/auth"
	"fashion-recommend/client"
	"fashion-recommend/database"
	"fashion-recommend/models"
	"fashion-recommend/traits"

	"github.com/gin-gonic/gin"
)

// Server API 服务器
type Server struct {
	gorseClient    *client.GorseClient
	authService    *auth.AuthService
	aiService      *ai.Service
	db             *database.DB
	traitExtractor *traits.Extractor
	gorseSync      *traits.GorseSync
	router         *gin.Engine
}

// NewServer 创建 API 服务器
func NewServer(gorseEndpoint, gorseAPIKey string, aiConfig ai.Config, db *database.DB) *Server {
	gorseClient := client.NewGorseClient(gorseEndpoint, gorseAPIKey)
	aiService := ai.NewService(aiConfig)

	s := &Server{
		gorseClient:    gorseClient,
		authService:    auth.NewAuthService(),
		aiService:      aiService,
		db:             db,
		traitExtractor: traits.NewExtractor(aiService, db),
		gorseSync:      traits.NewGorseSync(gorseClient, db),
		router:         gin.Default(),
	}
	s.setupRoutes()
	return s
}

// setupRoutes 设置路由
func (s *Server) setupRoutes() {
	api := s.router.Group("/api")
	{
		// 认证相关
		api.POST("/auth/register", s.register)
		api.POST("/auth/login", s.login)
		api.POST("/auth/logout", s.logout)
		api.GET("/auth/me", s.getCurrentUser)

		// AI 相关
		api.POST("/ai/chat", s.aiChat)
		api.POST("/ai/explain", s.aiExplainRecommendation)
		api.POST("/ai/style-advice", s.aiStyleAdvice)

		// Python agent proxy — forward to LangGraph agent on :8001
		api.POST("/ai/agent-chat", s.proxyToPythonAgent)
		api.POST("/ai/agent-resume", s.proxyToPythonAgent)

		// 评论相关
		api.POST("/comments", s.createComment)
		api.GET("/comments/:item_id", s.getComments)
		api.POST("/comments/:comment_id/like", s.likeComment)
		api.DELETE("/comments/:comment_id/like", s.unlikeComment)

		// 商品点赞相关
		api.POST("/products/:item_id/like", s.likeProduct)
		api.DELETE("/products/:item_id/like", s.unlikeProduct)
		api.GET("/products/:item_id/likes", s.getProductLikes)
		api.POST("/products/likes/batch", s.getBatchProductLikes)

		// 用户相关
		api.POST("/user", s.createUser)
		api.GET("/user/:user_id", s.getUser)
		api.POST("/users", s.createUsers)

		// 商品相关
		api.POST("/item", s.createItem)
		api.GET("/item/:item_id", s.getItem)
		api.POST("/items", s.createItems)

		// 反馈相关
		api.POST("/feedback", s.createFeedback)

		// 推荐相关
		api.GET("/recommend/:user_id", s.getRecommend)
		api.GET("/similar/:item_id", s.getSimilarItems)
	}

	// 健康检查
	s.router.GET("/health", s.healthCheck)

	// 静态文件服务 - 服务前端构建的文件
	s.router.Static("/assets", "./frontend/dist/assets")
	s.router.Static("/images", "./public/images")
	s.router.StaticFile("/", "./frontend/dist/index.html")

	// 所有未匹配的路由都返回 index.html（支持前端路由）
	s.router.NoRoute(func(c *gin.Context) {
		c.File("./frontend/dist/index.html")
	})
}

// createUser 创建用户
func (s *Server) createUser(c *gin.Context) {
	var user models.User
	if err := c.ShouldBindJSON(&user); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.gorseClient.InsertUser(user); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "user created successfully", "user_id": user.UserId})
}

// getUser 获取用户
func (s *Server) getUser(c *gin.Context) {
	userId := c.Param("user_id")

	user, err := s.gorseClient.GetUser(userId)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, user)
}

// createUsers 批量创建用户
func (s *Server) createUsers(c *gin.Context) {
	var users []models.User
	if err := c.ShouldBindJSON(&users); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.gorseClient.InsertUsers(users); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "users created successfully", "count": len(users)})
}

// createItem 创建商品
func (s *Server) createItem(c *gin.Context) {
	var item models.Item
	if err := c.ShouldBindJSON(&item); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.gorseClient.InsertItem(item); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "item created successfully", "item_id": item.ItemId})
}

// getItem 获取商品
func (s *Server) getItem(c *gin.Context) {
	itemId := c.Param("item_id")

	item, err := s.gorseClient.GetItem(itemId)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, item)
}

// createItems 批量创建商品
func (s *Server) createItems(c *gin.Context) {
	var items []models.Item
	if err := c.ShouldBindJSON(&items); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.gorseClient.InsertItems(items); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "items created successfully", "count": len(items)})
}

// createFeedback 创建反馈
func (s *Server) createFeedback(c *gin.Context) {
	var feedback []models.Feedback
	if err := c.ShouldBindJSON(&feedback); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := s.gorseClient.InsertFeedback(feedback); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "feedback created successfully", "count": len(feedback)})
}

// getRecommend 获取推荐
func (s *Server) getRecommend(c *gin.Context) {
	userId := c.Param("user_id")
	category := c.DefaultQuery("category", "")
	n, _ := strconv.Atoi(c.DefaultQuery("n", "10"))

	recommended, err := s.gorseClient.GetRecommend(userId, category, n)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Enrich each item for the frontend. Name and price come from the item's
	// Comment carrier (models.ItemCarrier); price_range and avg_rating are still
	// labels, just no longer under the similarity branch.
	type EnrichedItem struct {
		ItemId     string  `json:"item_id"`
		Score      float64 `json:"score"`
		Name       string  `json:"name"`
		Price      float64 `json:"price"`
		PriceRange string  `json:"price_range"`
		AvgRating  float64 `json:"avg_rating"`
	}
	enriched := make([]EnrichedItem, 0, len(recommended))
	for _, rec := range recommended {
		if rec.ItemId == "" {
			continue
		}
		item := EnrichedItem{ItemId: rec.ItemId, Score: rec.Score}
		if full, err := s.gorseClient.GetItem(rec.ItemId); err == nil {
			// 名字和价格来自 Comment 载体，不再来自标签。它们不是特征，放在
			// Labels 里会被两个模型都当特征计费 —— 而且是最坏的一类，因为两者
			// 都近乎唯一。详见 models.ItemCarrier。
			carrier := models.ParseItemCarrier(full.Comment)
			item.Name = carrier.Name
			if carrier.Price != nil {
				item.Price = *carrier.Price
			}
			// 这两个仍然是标签，只是移出了相似度分支（f）——它们是合法的 CTR
			// 特征，只是不该当相似度信号。
			item.PriceRange = full.Labels.PriceRange
			if r, err := strconv.ParseFloat(full.Labels.AvgRating, 64); err == nil {
				item.AvgRating = r
			}
		}
		enriched = append(enriched, item)
	}

	c.JSON(http.StatusOK, gin.H{
		"user_id": userId,
		"items":   enriched,
		"total":   len(enriched),
	})
}

// getSimilarItems 获取相似商品
func (s *Server) getSimilarItems(c *gin.Context) {
	itemId := c.Param("item_id")
	n, _ := strconv.Atoi(c.DefaultQuery("n", "10"))

	items, err := s.gorseClient.GetItemNeighbors(itemId, n)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"item_id": itemId,
		"items":   items,
		"total":   len(items),
	})
}

// healthCheck 健康检查
func (s *Server) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "fashion-recommend-api",
	})
}

// proxyToPythonAgent reverse-proxies /api/ai/agent-* requests to the
// LangGraph python agent running on PYTHON_AGENT_URL (default :8001).
func (s *Server) proxyToPythonAgent(c *gin.Context) {
	agentURL := os.Getenv("PYTHON_AGENT_URL")
	if agentURL == "" {
		agentURL = "http://localhost:8001"
	}
	target, err := url.Parse(agentURL)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "invalid python agent URL"})
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, e error) {
		c.JSON(http.StatusBadGateway, gin.H{"error": "python agent unavailable: " + e.Error()})
	}
	// Strip the response writer hijack so Gin doesn't double-write headers
	proxy.ModifyResponse = func(resp *http.Response) error {
		// Pass through as-is
		_ = io.Discard
		return nil
	}
	proxy.ServeHTTP(c.Writer, c.Request)
}

// Run 启动服务器
func (s *Server) Run(addr string) error {
	return s.router.Run(addr)
}

// GetRouter 获取路由器（用于测试）
func (s *Server) GetRouter() *gin.Engine {
	return s.router
}
