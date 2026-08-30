import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Heart, MessageCircle } from 'lucide-react'
import CommentDrawer from './CommentDrawer'
import apiService from '../services/api'

interface ProductCardProps {
  id: string
  name: string
  score: number
  price?: number
  priceRange?: string
  avgRating?: number
  onLike?: () => void
  onAddToCart?: () => void
}

export default function ProductCard({ id, name, price, priceRange, avgRating, onAddToCart }: ProductCardProps) {
  const [liked, setLiked] = useState(false)
  const [likeCount, setLikeCount] = useState(0)
  const [showComments, setShowComments] = useState(false)
  const [loading, setLoading] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const viewSent = useRef(false)

  useEffect(() => {
    loadLikeInfo()
  }, [id])

  // 曝光埋点：卡片一半以上进入视口、且停留超过 1 秒，才算一次 view。
  //
  // 两个约束都是必要的。没有面积阈值，快速滚过屏幕边缘就会算曝光；没有停留
  // 时间，一次滑到底会给整页商品刷满 view。Gorse 把 view 归在
  // read_feedback_types，噪声会直接稀释正反馈的相对权重。
  //
  // viewSent 用 ref 而不是 state：它只用来去重，不该触发重渲染，而且 state
  // 的异步更新会让同一张卡在 observer 连续回调里重复发送。
  useEffect(() => {
    viewSent.current = false
    const node = cardRef.current
    if (!node || typeof IntersectionObserver === 'undefined') return

    let timer: ReturnType<typeof setTimeout> | undefined
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !viewSent.current) {
          timer = setTimeout(() => {
            if (viewSent.current) return
            viewSent.current = true
            apiService.sendFeedback('view', id)
            observer.disconnect()
          }, 1000)
        } else if (timer) {
          clearTimeout(timer)
          timer = undefined
        }
      },
      { threshold: 0.5 }
    )
    observer.observe(node)

    return () => {
      if (timer) clearTimeout(timer)
      observer.disconnect()
    }
  }, [id])

  const loadLikeInfo = async () => {
    try {
      const username = localStorage.getItem('username') || 'guest'
      const data = await apiService.getProductLikes(id, username)
      setLiked(data.is_liked)
      setLikeCount(data.like_count)
    } catch (error) {
      console.error('Failed to load like info:', error)
    }
  }

  const handleLike = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (loading) return

    setLoading(true)
    try {
      const username = localStorage.getItem('username') || 'guest'
      
      if (liked) {
        const data = await apiService.unlikeProduct(id, username)
        setLiked(false)
        setLikeCount(data.like_count)
      } else {
        const data = await apiService.likeProduct(id, username)
        setLiked(true)
        setLikeCount(data.like_count)
      }
    } catch (error) {
      console.error('Failed to toggle like:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleComment = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowComments(true)
  }

  const handleAddToCart = (e: React.MouseEvent) => {
    e.stopPropagation()
    apiService.sendFeedback('add_to_cart', id)
    onAddToCart?.()
  }

  return (
    <>
      <motion.div
        ref={cardRef}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="group cursor-pointer h-full"
      >
        {/* Product Image Container */}
        <motion.div
          className="relative mb-6 overflow-hidden rounded-lg border-l-4 border-orange-500 bg-white shadow-lg"
          whileHover={{ y: -10, boxShadow: '0 20px 40px rgba(217, 119, 6, 0.2)' }}
          transition={{ duration: 0.3 }}
        >
          {/* Image Container */}
          <div className="aspect-[3/4] flex items-center justify-center overflow-hidden relative bg-gradient-to-br from-orange-100 to-red-100">
            <motion.img
              src={`/images/${id}.jpg`}
              alt={name}
              className="w-full h-full object-cover"
              initial={{ opacity: 0, scale: 1.1 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              onLoad={() => setImageLoaded(true)}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                const parent = e.currentTarget.parentElement;
                if (parent && !parent.querySelector('.img-fallback')) {
                  const span = document.createElement('span');
                  span.className = 'img-fallback text-sm font-serif text-orange-400 text-center px-4';
                  span.innerText = name || id;
                  parent.appendChild(span);
                }
              }}
            />

            {/* Image Loading Skeleton */}
            {!imageLoaded && (
              <div className="absolute inset-0 bg-gradient-to-r from-orange-200 via-red-200 to-orange-200 animate-pulse" />
            )}
          </div>

          {/* Like Button */}
          <motion.button
            onClick={handleLike}
            disabled={loading}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
            className="absolute top-4 right-4 p-2 bg-white border-2 border-orange-400 rounded-lg hover:bg-orange-50 transition-all disabled:opacity-50 z-10 shadow-md"
          >
            <Heart
              className={`h-4 w-4 transition-colors ${
                liked ? 'fill-red-500 text-red-500' : 'text-orange-400'
              }`}
            />
          </motion.button>

          {/* Like Count Badge */}
          {likeCount > 0 && (
            <motion.div
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="absolute top-4 left-4 px-3 py-1 bg-white border-2 border-orange-400 text-xs font-serif font-medium text-orange-600 rounded-lg shadow-md"
            >
              {likeCount}
            </motion.div>
          )}
        </motion.div>

        {/* Product Info */}
        <div className="pt-4 px-1">
          <h3 className="text-sm font-serif font-semibold text-amber-950 mb-2 tracking-tight">
            {name || `Product ${id}`}
          </h3>

          {avgRating ? (
            <p className="text-xs text-amber-900 mb-4 font-serif">
              ★ <span className="text-orange-600 font-semibold">{avgRating.toFixed(1)}</span>
              <span className="text-amber-700">/5</span>
            </p>
          ) : null}

          {/* Actions */}
          <div className="flex items-center justify-between">
            <span className="text-base font-serif font-semibold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
              {price
                ? `$${price.toFixed(2)}`
                : priceRange === 'budget' ? '$'
                : priceRange === 'mid' ? '$$'
                : priceRange === 'premium' ? '$$$'
                : '—'}
            </span>

            <div className="flex items-center space-x-3">
              {/* Comment Button */}
              <motion.button
                onClick={handleComment}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="p-1.5 border-2 border-orange-400 hover:bg-orange-50 rounded-lg transition-colors"
              >
                <MessageCircle className="h-4 w-4 text-orange-600" />
              </motion.button>

              {/* Add to Cart Button */}
              <motion.button
                onClick={handleAddToCart}
                whileHover={{ scale: 1.05, boxShadow: '0 10px 20px rgba(217, 119, 6, 0.3)' }}
                whileTap={{ scale: 0.95 }}
                className="px-4 py-1.5 bg-gradient-to-r from-orange-500 to-red-600 text-white text-xs font-serif font-medium rounded-lg hover:shadow-lg transition-all"
              >
                Add
              </motion.button>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Comment Drawer */}
      <CommentDrawer
        isOpen={showComments}
        onClose={() => setShowComments(false)}
        itemId={id}
        itemName={name || `Product ${id}`}
      />
    </>
  )
}
