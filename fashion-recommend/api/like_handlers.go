package api

import (
	"log"
	"net/http"
	"time"

	"fashion-recommend/models"

	"github.com/gin-gonic/gin"
)

// likeProduct 点赞商品
func (s *Server) likeProduct(c *gin.Context) {
	itemID := c.Param("item_id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Item ID is required"})
		return
	}

	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = "guest"
	}

	if err := s.db.LikeProduct(itemID, userID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to like product"})
		return
	}

	// 点赞同时写一条 favorite 反馈给 Gorse。config.toml 的
	// positive_feedback_types 包含 favorite，所以这条会进模型。
	//
	// 失败不影响响应：点赞本身已经落到 Postgres 了，Gorse 是次要目的地，
	// 让它把一次成功的点赞变成 500 是不对的。
	s.sendFeedback("favorite", userID, itemID)

	// 获取最新点赞数
	count, _ := s.db.GetProductLikeCount(itemID)

	c.JSON(http.StatusOK, gin.H{
		"message":    "Product liked successfully",
		"like_count": count,
	})
}

// sendFeedback 尽力向 Gorse 写一条反馈，失败只记日志。
//
// 注意取消点赞没有对应的撤销动作 —— GorseClient 目前没有删除反馈的方法，
// Gorse 侧的 favorite 会留着。这条链路的定位是打通管道，不是训练信号的来源
// （模型训练用的是 reco_interactions 的 train split），所以先不补。
func (s *Server) sendFeedback(feedbackType, userID, itemID string) {
	err := s.gorseClient.InsertFeedback([]models.Feedback{{
		FeedbackType: feedbackType,
		UserId:       userID,
		ItemId:       itemID,
		Timestamp:    time.Now(),
	}})
	if err != nil {
		log.Printf("写入 Gorse 反馈失败 (%s, user=%s, item=%s): %v",
			feedbackType, userID, itemID, err)
	}
}

// unlikeProduct 取消点赞商品
func (s *Server) unlikeProduct(c *gin.Context) {
	itemID := c.Param("item_id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Item ID is required"})
		return
	}

	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = "guest"
	}

	if err := s.db.UnlikeProduct(itemID, userID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to unlike product"})
		return
	}

	// 获取最新点赞数
	count, _ := s.db.GetProductLikeCount(itemID)

	c.JSON(http.StatusOK, gin.H{
		"message":    "Product unliked successfully",
		"like_count": count,
	})
}

// getProductLikes 获取商品点赞信息
func (s *Server) getProductLikes(c *gin.Context) {
	itemID := c.Param("item_id")
	if itemID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Item ID is required"})
		return
	}

	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = "guest"
	}

	count, err := s.db.GetProductLikeCount(itemID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get like count"})
		return
	}

	isLiked, err := s.db.IsProductLiked(itemID, userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to check like status"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"item_id":    itemID,
		"like_count": count,
		"is_liked":   isLiked,
	})
}

// getBatchProductLikes 批量获取商品点赞信息
func (s *Server) getBatchProductLikes(c *gin.Context) {
	var req struct {
		ItemIDs []string `json:"item_ids" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	userID := c.GetHeader("X-User-ID")
	if userID == "" {
		userID = "guest"
	}

	likeInfo, err := s.db.GetProductsLikeInfo(req.ItemIDs, userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get like info"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"likes": likeInfo,
	})
}
