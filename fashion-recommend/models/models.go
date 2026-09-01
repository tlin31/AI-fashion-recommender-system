package models

import (
	"encoding/json"
	"time"
)

// User 用户模型。
//
// 字段名必须和 Gorse REST API 的线格式完全一致（UserId / Labels / ...），
// 跟下面的 Item 一样。原来这里是 snake_case 的 user_id / labels，后果是
// POST /api/user(s) 发出去的 user_id 对不上 Gorse 的 UserId 字段 ——
// encoding/json 反序列化对大小写不敏感，所以 labels 侥幸能对上 Labels，
// 但带下划线的 user_id 对不上，于是 Gorse 收到的每个用户 UserId 都是空串，
// 全部写进同一行空 ID 记录。trait 同步因此从来没有真正落到过任何用户身上。
//
// 这个 bug 躲过了 server_test.go，因为那里的 mock Gorse 用同一个 struct
// 编码响应、再用同一个 struct 解码，tag 不一致在往返中被抵消了。参见
// models_wire_test.go —— 那里断言的是字面 JSON key，不是往返。
type User struct {
	UserId    string   `json:"UserId"`
	Labels    []string `json:"Labels"`
	Comment   string   `json:"Comment,omitempty"`
	Subscribe []string `json:"Subscribe,omitempty"`
}

// ItemLabels 是 Gorse item 标签的 map 形式。
//
// Gorse 的 Labels 字段类型是 any，扁平数组和嵌套 map 都收，但两种形式到达的
// 消费者不同：
//
//	tags item-to-item 只读一个分支，由 expr 表达式指名
//	  （config.toml 里 column = "item.Labels.f"）。logics 的 flatten
//	  （item_to_item.go:379）接受 []dataset.ID，所以 f 下的字符串数组原样到达，
//	  而 f 之外的任何东西它都看不见。
//	CTR 模型读整张 map（ctr.ConvertLabels，model/ctr/data.go:39），
//	  把 {"brand": "Nike"} 摊平成特征名 brand.Nike。
//
// 这个分叉就是全部的修复。扁平 schema 下实测 item-to-item 的分数落在
// 0.500689–0.510655，单个 top-10 内跨度中位数 0.000339 —— 全部挤在 [0,1] 的
// 1% 里。原因不在相似度函数：能区分的标签（item_name、brand，近乎唯一）
// 从不匹配，能匹配的标签（price_range 有 80% 是 mid、avg_rating 25 个粗桶）
// 不区分。两件任意商品共享的就是 price_range:mid 加 avg_rating:4.5，
// 而两件都被打 4.5 分的商品在任何购物者认得的意义上都不相似。
//
// Brand / PriceRange / AvgRating 没有被删掉，只是移出了 f。它们是合法的 CTR
// 特征（FM 能学 brand 的 embedding），只是从来就不该当相似度信号用。
// 「移出相似度路径」和「删除」是两句不同的话。
type ItemLabels struct {
	// Features 是唯一走相似度路径的分支：type: / cat: / style: / color: /
	// occasion: / material:
	Features   []string `json:"f"`
	Brand      string   `json:"brand,omitempty"`
	PriceRange string   `json:"price_range,omitempty"`
	// AvgRating 是字符串而不是数字，这是刻意的。map 键下的 JSON 数值两个模型
	// 都读不到 —— ctr.convertLabels 没有 float64 分支（gorm 的 json
	// serializer 解成 float64，不是 json.Number），flatten 也直接忽略数值叶子。
	// 写成数字会得到一个看起来像特征、实际完全惰性的字段。
	AvgRating string `json:"avg_rating,omitempty"`
}

// UnmarshalJSON 同时接受扁平数组和 map 两种形式。
//
// 重新 seed 目录要跑几十分钟，期间 Gorse 里两种形式的 item 会共存；解码器只认
// 一种的话，API 会在这段时间里对着一半的商品报错。扁平数组按它历史上的语义
// 处理：全部当作特征。
func (l *ItemLabels) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '[' {
		var flat []string
		if err := json.Unmarshal(b, &flat); err != nil {
			return err
		}
		l.Features = flat
		return nil
	}
	type alias ItemLabels // 避免递归调用自己
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*l = ItemLabels(a)
	return nil
}

// ItemCarrier 是随 Item.Comment 走的载体数据：前端要、但任何模型都不该看见的东西。
//
// item_name 和 price 原本是标签，那是放错了字段。它们不是特征 —— 存在的理由是
// api/server.go 的 enrichment 要给前端填名字和价格 —— 但待在 Labels 里就会被
// 当特征计费。在 tags 路径上它们还是最坏的一类：两者都近乎唯一，IDF 各约
// log(95335) ≈ 11.5，而一个真正共享的 style 标签只有约 1.5；而距离函数
// （item_to_item.go:337）除以的是该商品**全部**标签权重和的平方根。两个载体
// 给分母加了约 23，真特征几乎推不动它。
//
// 选 Comment 是因为它被证实是惰性的：在 logics/ master/ model/ dataset/ 里
// grep `.Comment` 没有任何结果。没有任何推荐器会读的存储，正是载体想要的。
type ItemCarrier struct {
	Name  string   `json:"name"`
	Price *float64 `json:"price"`
	Desc  string   `json:"desc"`
}

// ParseItemCarrier 解析 Comment。解析失败不是错误：Comment 历史上存的是纯
// 描述文本，而且它本来就是自由字段。这种情况把整段当描述，名字和价格留空。
func ParseItemCarrier(comment string) ItemCarrier {
	var c ItemCarrier
	if err := json.Unmarshal([]byte(comment), &c); err != nil {
		return ItemCarrier{Desc: comment}
	}
	return c
}

// Item 商品模型
type Item struct {
	ItemId     string     `json:"ItemId"`
	IsHidden   bool       `json:"IsHidden"`
	Categories []string   `json:"Categories"`
	Labels     ItemLabels `json:"Labels"`
	Comment    string     `json:"Comment,omitempty"`
	Timestamp  time.Time  `json:"Timestamp"`
}

// Feedback 反馈模型。
//
// 和 User 一样，字段名必须匹配 Gorse 的线格式。storage/data/database.go 的
// Feedback / FeedbackKey 没有 json tag，所以 Gorse 认的是 Go 字段名本身：
// FeedbackType / UserId / ItemId / Value / Timestamp。原来这里的 snake_case
// 带下划线，对不上，反馈写进去时三个 key 全是空的。
type Feedback struct {
	FeedbackType string    `json:"FeedbackType"`
	UserId       string    `json:"UserId"`
	ItemId       string    `json:"ItemId"`
	Value        float64   `json:"Value"`
	Timestamp    time.Time `json:"Timestamp"`
	Comment      string    `json:"Comment,omitempty"`
}

// RecommendRequest 推荐请求
type RecommendRequest struct {
	UserId   string `json:"user_id"`
	N        int    `json:"n"`
	Category string `json:"category,omitempty"`
	Offset   int    `json:"offset,omitempty"`
}

// RecommendResponse 推荐响应
type RecommendResponse struct {
	Items []RecommendItem `json:"items"`
	Total int             `json:"total"`
}

// RecommendItem 推荐商品项
type RecommendItem struct {
	ItemId     string   `json:"item_id"`
	Score      float64  `json:"score"`
	Categories []string `json:"categories,omitempty"`
}

// FashionUserBuilder 时尚用户构建器
type FashionUserBuilder struct {
	user User
}

func NewFashionUser(userId string) *FashionUserBuilder {
	return &FashionUserBuilder{
		user: User{
			UserId: userId,
			Labels: make([]string, 0),
		},
	}
}

func (b *FashionUserBuilder) WithGender(gender string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "gender:"+gender)
	return b
}

func (b *FashionUserBuilder) WithAgeGroup(ageGroup string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "age_group:"+ageGroup)
	return b
}

func (b *FashionUserBuilder) WithStyle(styles ...string) *FashionUserBuilder {
	for _, style := range styles {
		b.user.Labels = append(b.user.Labels, "style:"+style)
	}
	return b
}

// WithPricePreference 写入 price_range: 前缀 —— 和 item 侧同名。
// priceRange 应取 item 侧的词表（budget / mid / premium），不是自造的
// mid-range / high-end / luxury 之类。
func (b *FashionUserBuilder) WithPricePreference(priceRange string) *FashionUserBuilder {
	b.user.Labels = append(b.user.Labels, "price_range:"+priceRange)
	return b
}

// WithFavoriteBrands 写入 brand: 前缀 —— 和 item 侧同名，原来的
// favorite_brand: 让同一个品牌在两侧永远是两个不同的字符串。
func (b *FashionUserBuilder) WithFavoriteBrands(brands ...string) *FashionUserBuilder {
	for _, brand := range brands {
		b.user.Labels = append(b.user.Labels, "brand:"+brand)
	}
	return b
}

func (b *FashionUserBuilder) WithComment(comment string) *FashionUserBuilder {
	b.user.Comment = comment
	return b
}

func (b *FashionUserBuilder) Build() User {
	return b.user
}

// FashionItemBuilder 时尚商品构建器
type FashionItemBuilder struct {
	item Item
}

func NewFashionItem(itemId string) *FashionItemBuilder {
	return &FashionItemBuilder{
		item: Item{
			ItemId:     itemId,
			IsHidden:   false,
			Categories: make([]string, 0),
			Labels:     ItemLabels{Features: make([]string, 0)},
			Timestamp:  time.Now(),
		},
	}
}

func (b *FashionItemBuilder) WithCategories(categories ...string) *FashionItemBuilder {
	b.item.Categories = append(b.item.Categories, categories...)
	return b
}

func (b *FashionItemBuilder) WithBrand(brand string) *FashionItemBuilder {
	b.item.Labels.Brand = brand
	return b
}

func (b *FashionItemBuilder) WithSeason(season string) *FashionItemBuilder {
	b.item.Labels.Features = append(b.item.Labels.Features, "season:"+season)
	return b
}

func (b *FashionItemBuilder) WithStyle(styles ...string) *FashionItemBuilder {
	for _, style := range styles {
		b.item.Labels.Features = append(b.item.Labels.Features, "style:"+style)
	}
	return b
}

func (b *FashionItemBuilder) WithColor(colors ...string) *FashionItemBuilder {
	for _, color := range colors {
		b.item.Labels.Features = append(b.item.Labels.Features, "color:"+color)
	}
	return b
}

func (b *FashionItemBuilder) WithMaterial(material string) *FashionItemBuilder {
	b.item.Labels.Features = append(b.item.Labels.Features, "material:"+material)
	return b
}

func (b *FashionItemBuilder) WithPriceRange(priceRange string) *FashionItemBuilder {
	b.item.Labels.PriceRange = priceRange
	return b
}

func (b *FashionItemBuilder) WithOccasion(occasions ...string) *FashionItemBuilder {
	for _, occasion := range occasions {
		b.item.Labels.Features = append(b.item.Labels.Features, "occasion:"+occasion)
	}
	return b
}

func (b *FashionItemBuilder) WithComment(comment string) *FashionItemBuilder {
	b.item.Comment = comment
	return b
}

func (b *FashionItemBuilder) WithTimestamp(timestamp time.Time) *FashionItemBuilder {
	b.item.Timestamp = timestamp
	return b
}

func (b *FashionItemBuilder) Hidden() *FashionItemBuilder {
	b.item.IsHidden = true
	return b
}

func (b *FashionItemBuilder) Build() Item {
	return b.item
}
