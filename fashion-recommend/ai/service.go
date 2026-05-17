package ai

import (
	"context"
	"fmt"

	openai "github.com/sashabaranov/go-openai"
)

// Config AI 配置
type Config struct {
	APIKey      string
	BaseURL     string
	Model       string
	MaxTokens   int
	Temperature float32
}

// Service AI 服务
type Service struct {
	client *openai.Client
	config Config
}

// NewService 创建 AI 服务
func NewService(config Config) *Service {
	clientConfig := openai.DefaultConfig(config.APIKey)
	clientConfig.BaseURL = config.BaseURL
	
	return &Service{
		client: openai.NewClientWithConfig(clientConfig),
		config: config,
	}
}

// Chat 对话
func (s *Service) Chat(ctx context.Context, messages []openai.ChatCompletionMessage) (string, error) {
	req := openai.ChatCompletionRequest{
		Model:       s.config.Model,
		Messages:    messages,
		MaxTokens:   s.config.MaxTokens,
		Temperature: s.config.Temperature,
	}

	resp, err := s.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("AI 对话失败: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("AI 未返回响应")
	}

	return resp.Choices[0].Message.Content, nil
}

// ExplainRecommendation 解释推荐理由
func (s *Service) ExplainRecommendation(ctx context.Context, username string, itemID string, userPreferences []string) (string, error) {
	// 原中文 prompt（保留供参考）:
	// systemPrompt := "你是一个时尚品牌推荐系统的 AI 助手，擅长解释为什么推荐某个商品给用户。请用简洁、友好的语言解释推荐理由。"
	systemPrompt := "You are an AI assistant for a fashion brand recommendation system, skilled at explaining why a product is recommended to a user. Please explain the recommendation reason in a concise and friendly tone."

	// 原中文 prompt（保留供参考）:
	// userPrompt := fmt.Sprintf(
	// 	"用户 %s 的偏好是：%v\n我们推荐了商品 %s 给这位用户。请用1-2句话解释为什么推荐这个商品。",
	// 	username, userPreferences, itemID,
	// )
	userPrompt := fmt.Sprintf(
		"User %s has the following preferences: %v\nWe recommended item %s to this user. Please explain in 1-2 sentences why this item was recommended.",
		username,
		userPreferences,
		itemID,
	)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: userPrompt,
		},
	}

	return s.Chat(ctx, messages)
}

// GenerateStyleAdvice 生成穿搭建议
func (s *Service) GenerateStyleAdvice(ctx context.Context, items []string, occasion string) (string, error) {
	// 原中文 prompt（保留供参考）:
	// systemPrompt := "你是一个专业的时尚造型师，擅长为用户提供穿搭建议。"
	systemPrompt := "You are a professional fashion stylist, skilled at providing outfit and styling advice to users."

	// 原中文 prompt（保留供参考）:
	// userPrompt := fmt.Sprintf(
	// 	"用户选择了这些商品：%v\n场合是：%s\n请提供简洁的穿搭建议和搭配技巧。",
	// 	items, occasion,
	// )
	userPrompt := fmt.Sprintf(
		"The user has selected these items: %v\nThe occasion is: %s\nPlease provide concise outfit advice and styling tips.",
		items,
		occasion,
	)

	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
		{
			Role:    openai.ChatMessageRoleUser,
			Content: userPrompt,
		},
	}

	return s.Chat(ctx, messages)
}

// ChatWithAssistant 与 AI 助手对话
func (s *Service) ChatWithAssistant(ctx context.Context, userMessage string, conversationHistory []openai.ChatCompletionMessage) (string, error) {
	// 原中文 prompt（保留供参考）:
	// systemPrompt := "你是一个时尚品牌推荐系统的 AI 助手，名字叫「时尚小助手」。你可以帮助用户了解时尚趋势、推荐商品、提供穿搭建议。请用友好、专业的语气回答用户的问题。"
	systemPrompt := "You are an AI assistant for a fashion brand recommendation system, named \"Fashion Curator\". You can help users explore fashion trends, discover products, and get outfit advice. Always respond in a friendly and professional tone in English."
	
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleSystem,
			Content: systemPrompt,
		},
	}
	
	// 添加历史对话
	messages = append(messages, conversationHistory...)
	
	// 添加当前用户消息
	messages = append(messages, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: userMessage,
	})

	return s.Chat(ctx, messages)
}
